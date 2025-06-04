from flask import Flask, request, jsonify
import time
import random

app = Flask(__name__)

# Simulate different response times
MIN_RESPONSE_TIME = 0.1  # seconds
MAX_RESPONSE_TIME = 0.5  # seconds


@app.route("/completion", methods=["GET"])
def completion():
    # Get the prompt from query parameters
    prompt = request.args.get("prompt", "")

    # Simulate processing time
    time.sleep(random.uniform(MIN_RESPONSE_TIME, MAX_RESPONSE_TIME))

    # Generate a dummy response
    response = {
        "content": f"Test response for prompt: {prompt[:50]}...",
        "stop": True,
        "timings": {"prompt_n": len(prompt), "time_total": random.uniform(0.1, 0.5)},
    }

    return jsonify(response)


if __name__ == "__main__":
    print("Starting test server on http://localhost:8080")
    print("Use Ctrl+C to stop the server")
    app.run(host="0.0.0.0", port=8080)
