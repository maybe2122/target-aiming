from deliverables.sim_inference_bundle.client.vla_client import VLAClient


def main():
    client = VLAClient("http://127.0.0.1:8000")

    result = client.predict_path(
        "/path/to/frame.jpg",
        instruction="Track the vehicle in the scene.",
        predict_type="grounding_action",
    )

    print("health:", client.health())
    print("action:", result.get("action"))
    print("latency_ms:", result.get("latency_ms"))


if __name__ == "__main__":
    main()
