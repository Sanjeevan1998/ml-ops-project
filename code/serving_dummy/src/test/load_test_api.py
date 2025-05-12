import asyncio
import aiohttp
import time
import os
import argparse
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


DEFAULT_API_URL = "http://129.114.24.228:8000/search_combined" # this is my kvm's floating ip


DEFAULT_PDF_PATH = os.path.join("real_pdfs", "Glover v. State.PDF") 
DEFAULT_CONCURRENT_REQUESTS = 10
DEFAULT_TOTAL_REQUESTS = 100
DEFAULT_TOP_K = 1

async def send_request(session: aiohttp.ClientSession, url: str, pdf_path: str, top_k: int, request_num: int) -> Dict[str, Any]:
    """Sends a single POST request with a file upload."""
    data = aiohttp.FormData()
    data.add_field('top_k', str(top_k))
    
    try:

        with open(pdf_path, 'rb') as f:
            data.add_field('query_file',
                           f,
                           filename=os.path.basename(pdf_path),
                           content_type='application/pdf')
            
            start_time = time.monotonic()
            async with session.post(url, data=data) as response:
                response_text = await response.text() # 
                end_time = time.monotonic()
                latency = end_time - start_time
                
                if response.status == 200:
                    logger.debug(f"Request {request_num} successful in {latency:.4f}s. Status: {response.status}")
                    return {"status": "success", "http_status": response.status, "latency": latency, "error": None}
                else:
                    logger.warning(f"Request {request_num} failed. Status: {response.status}, Latency: {latency:.4f}s, Response: {response_text[:200]}")
                    return {"status": "failure", "http_status": response.status, "latency": latency, "error": response_text[:200]}
    except FileNotFoundError:
        logger.error(f"Request {request_num} failed: PDF file not found at {pdf_path}")
        return {"status": "failure", "http_status": None, "latency": 0, "error": "PDFFileNotFound"}
    except aiohttp.ClientConnectorError as e:
        logger.error(f"Request {request_num} failed: Connection error - {e}")
        return {"status": "failure", "http_status": None, "latency": 0, "error": f"ClientConnectorError: {e}"}
    except Exception as e:
        logger.error(f"Request {request_num} failed with unexpected error: {e}")
        return {"status": "failure", "http_status": None, "latency": 0, "error": str(e)}

async def run_load_test(url: str, pdf_path: str, num_concurrent: int, total_requests: int, top_k: int):
    """Runs the load test with the given parameters."""
    logger.info(f"Starting load test: URL='{url}', PDF='{pdf_path}', Concurrent={num_concurrent}, Total={total_requests}, TopK={top_k}")
    
    if not os.path.exists(pdf_path):
        logger.error(f"Critical: Sample PDF file '{pdf_path}' not found. Aborting load test.")
        return

    tasks = []
    results: List[Dict[str, Any]] = []
    
    async with aiohttp.ClientSession() as session:
        overall_start_time = time.monotonic()
        
        semaphore = asyncio.Semaphore(num_concurrent) 

        async def R_worker(req_num):
            async with semaphore:
                res = await send_request(session, url, pdf_path, top_k, req_num)
                results.append(res)

        for i in range(total_requests):
            tasks.append(asyncio.create_task(R_worker(i + 1)))
        
        await asyncio.gather(*tasks)
        
        overall_end_time = time.monotonic()

    total_duration = overall_end_time - overall_start_time
    successful_requests = sum(1 for r in results if r["status"] == "success")
    failed_requests = total_requests - successful_requests
    
    logger.info("Load Test Summary:")
    logger.info(f"  Total requests sent: {total_requests}")
    logger.info(f"  Successful requests: {successful_requests}")
    logger.info(f"  Failed requests: {failed_requests}")
    logger.info(f"  Total time taken: {total_duration:.4f} seconds")
    if total_duration > 0:
        rps = total_requests / total_duration
        logger.info(f"  Requests per second (RPS): {rps:.4f}")
    
    if results:
        latencies = [r["latency"] for r in results if r["status"] == "success" and r["latency"] is not None]
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            latencies.sort()
            p95_latency = latencies[int(len(latencies) * 0.95)] if len(latencies) > 19 else (latencies[-1] if latencies else 0)

            logger.info(f"  Avg Latency (successful): {avg_latency:.4f}s")
            logger.info(f"  Min Latency (successful): {min_latency:.4f}s")
            logger.info(f"  Max Latency (successful): {max_latency:.4f}s")
            logger.info(f"  P95 Latency (successful): {p95_latency:.4f}s")
        else:
            logger.info("  No successful requests with latency data to report.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="API Load Tester")
    parser.add_argument("--url", type=str, default=DEFAULT_API_URL, help="API endpoint URL")
    parser.add_argument("--pdf_path", type=str, default=DEFAULT_PDF_PATH, help="Path to the PDF file for uploads")
    parser.add_argument("--concurrent", "-c", type=int, default=DEFAULT_CONCURRENT_REQUESTS, help="Number of concurrent requests")
    parser.add_argument("--total_requests", "-n", type=int, default=DEFAULT_TOTAL_REQUESTS, help="Total number of requests to send")
    parser.add_argument("--top_k", "-k", type=int, default=DEFAULT_TOP_K, help="Value for top_k parameter")
    
    args = parser.parse_args()

    if "<VM_FLOATING_IP>" in args.url:
        print("Error: Please replace <VM_FLOATING_IP> in the script's DEFAULT_API_URL or provide --url argument.")
        exit(1)
        
    asyncio.run(run_load_test(args.url, args.pdf_path, args.concurrent, args.total_requests, args.top_k))