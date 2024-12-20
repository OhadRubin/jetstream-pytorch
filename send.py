import sys
sys.path.append('/jetstream-pytorch/deps/JetStream')
import asyncio
import grpc
from jetstream.core.proto import jetstream_pb2
from jetstream.core.proto import jetstream_pb2_grpc
from jetstream.external_tokenizers.llama3 import llama3_tokenizer

from transformers import AutoTokenizer
import tiktoken

async def send_single_request(
    server_address: str,
    port: int,
    tokenizer_path: str,
    prompt: str,
    max_output_tokens: int = 20,
):
    """Sends a single request to the JetStream server and returns the response."""
    api_url = f"{server_address}:{port}"
    tokenizer = llama3_tokenizer.Tokenizer(tokenizer_path)
    token_ids = tokenizer.encode(prompt)
    request = jetstream_pb2.DecodeRequest(
        token_content=jetstream_pb2.DecodeRequest.TokenContent(token_ids=token_ids),
        max_tokens=max_output_tokens,
    )

    try:
        async with grpc.aio.insecure_channel(api_url) as channel:
            stub = jetstream_pb2_grpc.OrchestratorStub(channel)
            response_stream = stub.Decode(request)
            generated_token_ids = []
            async for response in response_stream:
                generated_token_ids.extend(response.stream_content.samples[0].token_ids)
            generated_text = tokenizer.decode(generated_token_ids)
            return generated_text
    except grpc.RpcError as e:
        print(f"Error during gRPC call: {e}")
        return None
async def main(
    server_address: str = "localhost",
    port: int = 9000,
    tokenizer_path: str = "/jetstream-pytorch/deps/JetStream/jetstream/tests/engine/third_party/llama3/tokenizer.model",
    prompt_text: str = "Write a short poem about a cat.",
    max_tokens: int = 30,
):
    """Main function to define server details and send the request.
    
    Args:
        server_address: Server address to connect to. Defaults to localhost.
        port: Server port to connect to. Defaults to 9000.
        tokenizer_path: Path to tokenizer model. Defaults to gemma-7b-it.
        prompt_text: Text prompt to generate from. Defaults to cat poem prompt.
        max_tokens: Maximum tokens to generate. Defaults to 30.
    """

    response = await send_single_request(
        server_address, port, tokenizer_path, prompt_text, max_tokens
    )

    if response:
        print(f"Generated response: {response}")
    else:
        print("Request failed.")
if __name__ == "__main__":
    import fire
    fire.Fire(main)