"""Live proxy tests. Start the server with scripts\\start.bat first."""
import httpx

PROXY_URL = "http://localhost:8000"


def test_health():
    resp = httpx.get(f"{PROXY_URL}/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    print("  health ok")


def test_stats():
    resp = httpx.get(f"{PROXY_URL}/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_chunks" in data
    print(f"  stats: {data['total_chunks']} chunks indexed")


def test_chat():
    resp = httpx.post(
        f"{PROXY_URL}/v1/chat/completions",
        json={
            "model": "deepseek-reasoner",
            "messages": [{"role": "user", "content": "What does this codebase do?"}],
            "stream": False,
        },
        timeout=120,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "choices" in data
    preview = data["choices"][0]["message"]["content"][:200]
    print(f"  chat ok, preview: {preview}")


if __name__ == "__main__":
    print("Testing proxy server (must be running)...")
    test_health()
    test_stats()
    test_chat()
    print("\nAll proxy tests passed.")
