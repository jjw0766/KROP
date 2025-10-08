import os
import argparse
import asyncio
import time
from dotenv import load_dotenv

import pandas as pd
from tqdm.asyncio import tqdm as async_tqdm
from openai import AsyncOpenAI, RateLimitError, APIError

# src.data.dataset 모듈이 필요합니다. 
# 원본 코드와 동일한 데이터 로더를 사용한다고 가정합니다.
from src.data.dataset import get_test_dataloader


def create_async_client(provider: str, api_key: str | None) -> AsyncOpenAI:
    """지정된 프로바이더에 대한 비동기 API 클라이언트를 생성합니다."""
    if provider == "openai":
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OpenAI API key가 없습니다. --api_key 인자로 전달하거나 .env 파일에 OPENAI_API_KEY를 설정해주세요.")
        return AsyncOpenAI(api_key=key)
    
    elif provider == "deepseek":
        key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise ValueError("DeepSeek API key가 없습니다. --api_key 인자로 전달하거나 .env 파일에 DEEPSEEK_API_KEY를 설정해주세요.")
        return AsyncOpenAI(api_key=key, base_url="https://api.deepseek.com/v1")
    
    else:
        raise ValueError(f"지원하지 않는 API 프로바이더입니다: {provider}")


async def get_api_correction(
    client: AsyncOpenAI, 
    model: str, 
    noisy_sentence: str,
    retries: int = 3,
    delay: int = 60
) -> str:
    """API를 호출하여 문장을 교정합니다."""
    system_prompt = "You are a helpful assistant that deobfuscate errors in sentences. Please return only the corrected sentence, without any additional explanation or introductory text."
    for attempt in range(retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": noisy_sentence},
                ],
                # temperature=0.0
            )
            corrected_sentence = response.choices[0].message.content.strip()
            created_time = response.created # The Unix timestamp (in seconds) of when the chat completion was created.
            end_time = time.time()
            time_taken = end_time - created_time
            return (corrected_sentence, time_taken)
        except RateLimitError as e:
            if attempt < retries - 1:
                print(f"Rate limit exceeded. Retrying in {delay} seconds... ({attempt + 1}/{retries})")
                await asyncio.sleep(delay)
            else:
                print(f"Rate limit error after {retries} retries: {e}")
                return f"ERROR: Rate limit exceeded - {noisy_sentence}"
        except APIError as e:
            if attempt < retries - 1:
                print(f"API error occurred: {e}. Retrying in {delay} seconds... ({attempt + 1}/{retries})")
                await asyncio.sleep(delay)
            else:
                print(f"API error after {retries} retries: {e}")
                return f"ERROR: API Error - {noisy_sentence}"
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return f"ERROR: Unexpected error - {noisy_sentence}"


async def main(args):
    """메인 비동기 추론 함수"""
    load_dotenv()
    
    # API 클라이언트 초기화
    try:
        client = create_async_client(args.api_provider, args.api_key)
    except ValueError as e:
        print(f"API 클라이언트 초기화 오류: {e}")
        return
    except Exception as e:
        print(f"클라이언트 초기화 중 예상치 못한 오류가 발생했습니다: {e}")
        return

    # 테스트 데이터 로더 (YAML 설정 없이 직접 인자 사용)
    test_dl = get_test_dataloader(
        args.dataset_name,
        batch_size=args.batch_size,
        select=args.test_dataset_select
    )

    # 결과를 저장할 리스트
    all_predictions, all_times, all_inputs, all_true, all_categories = [], [], [], [], []

    # tqdm을 사용하여 비동기 작업 진행률 표시
    progress_bar = async_tqdm(total=len(test_dl), desc="Processing batches")

    for batch in test_dl:
        start_time = time.time()
        
        # 현재 배치의 문장들로 비동기 작업 생성
        tasks = [
            get_api_correction(client, args.model, sentence)
            for sentence in batch["sentence_noisy"]
        ]
        
        # 생성된 작업들을 동시에 실행하고 결과 수집
        prediction_times = await asyncio.gather(*tasks)

        for predictions, time_taken in prediction_times:
            all_times.append(time_taken)
        predictions = [pred for pred, _ in prediction_times]

        
        # 결과 저장
        all_predictions.extend(predictions)
        all_inputs.extend(batch["sentence_noisy"])
        all_true.extend(batch["sentence"])
        
        category = batch.get("category") or ["none"] * len(batch["sentence"])
        all_categories.extend(category)
        progress_bar.update(1)
    
    progress_bar.close()

    # 결과 데이터프레임 생성
    result_df = pd.DataFrame({
        "input": all_inputs,
        "pred": all_predictions,
        "true": all_true,
        "category": all_categories,
        "all_times":all_times
    })

    # 결과 저장
    os.makedirs('results', exist_ok=True)
    result_df.to_csv(f"results/{args.model}-{args.dataset_name.split('/')[1]}.csv", index=False)


def setup_parser():
    """스크립트 인자 파서 설정"""
    parser = argparse.ArgumentParser(description="Inference with OpenAI/DeepSeek API (async)")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="사용할 데이터셋의 이름 (예: 'jfleg', 'conll2003')")
    parser.add_argument("--api_provider", type=str, default="openai", choices=["openai", "deepseek"],
                        help="사용할 API 프로바이더 ('openai' 또는 'deepseek')")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API 키. 제공되지 않으면 .env 파일에서 로드 (OPENAI_API_KEY or DEEPSEEK_API_KEY).")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo",
                        help="추론에 사용할 모델 이름 (예: gpt-3.5-turbo, deepseek-chat)")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="동시 API 요청 수")
    parser.add_argument("--test_dataset_select", type=int, default=-1,
                        help="테스트 데이터셋에서 사용할 샘플 수 (기본값: -1, 전체 사용)")

    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    asyncio.run(main(args))