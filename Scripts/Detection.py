import time
import base64
import cv2
import numpy as np
from PIL import Image
from inference_sdk import InferenceHTTPClient
from threading import Thread
from queue import Queue

global_counter = 0

def run_inference_batch(base64_images: list, client: InferenceHTTPClient):
    """
    Runs inference on a batch of images using the detection workflow.
    """
    try:
        print(f"Workflow started for batch of {len(base64_images)} images")

        start_time = time.time()
        # Run the inference workflow for the batch

        results_total = [] 
        for img in base64_images:
            results = client.run_workflow(
                workspace_name="suavcoco",
                workflow_id="combined-models",
                images={"image": img}
            )
            results_total.append(results)

        end_time = time.time()
        print(f"Workflow finished. Execution time: {end_time - start_time:.2f} seconds")

        return results_total

    except Exception as e:
        print(f"Inference error: {e}")
        return None


def geomatics_calculation(detections) -> None:
    """
    Placeholder for geomatics calculations using the output image.
    """
    global global_counter
    global_counter += 1
    try:
        print("Performing geomatics calculations...")
        # Simulate actual geomatics calculations
        time.sleep(.4)  # Simulating processing time
        print(f"Geomatics calculations complete")
    except Exception as e:
        print(f"Geomatics calculation error: {e}")


def inference_worker(image_queue: Queue, detection_queue: Queue, client: InferenceHTTPClient, batch_size=12):
    """
    Worker thread to process images in batches and run inference.
    """
    while True:
        batch = []
        while len(batch) < batch_size and not image_queue.empty():
            img = image_queue.get()
            if img is None:
                break

            # Convert PIL Image to OpenCV format
            cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
           
            # Convert the resized image to base64 format
            _, buffer = cv2.imencode('.png', cv_image)
            base64_image = base64.b64encode(buffer).decode('utf-8')

            batch.append(base64_image)
            image_queue.task_done()

        if not batch:
            break
        
        results = run_inference_batch(batch, client)

        if results is not None:
            for result in results:
                print(result)
                for detection in result[0]['consensus_predictions']['predictions']:
                    detection_queue.put(detection)


def geomatics_worker(detection_queue: Queue):
    """
    Worker thread to process detections and perform geomatics calculations.
    """
    while True:
        detections = detection_queue.get()
        if detections is None:
            break

        try:
            geomatics_calculation(detections)
        finally:
            detection_queue.task_done()


if __name__ == "__main__":
    # ---- Add Images Here ---- (Random images for testing threads)
    image_paths = [
        "../Doc-Resources/stopersimg.webp",
        "../Doc-Resources/testimg.png",
        "../Doc-Resources/stopersimg.webp",
        "../Doc-Resources/testimg.png",
        "../Doc-Resources/stopersimg.webp",
        "../Doc-Resources/testimg.png",
        "../Doc-Resources/stopersimg.webp",
        "../Doc-Resources/testimg.png",
        "../Doc-Resources/stopersimg.webp",
        "../Doc-Resources/testimg.png",
        "../Doc-Resources/stopersimg.webp",
        "../Doc-Resources/testimg.png",
        "../Doc-Resources/stopersimg.webp",
        "../Doc-Resources/testimg.png",
        "../Doc-Resources/stopersimg.webp",
        "../Doc-Resources/testimg.png",
        "../Doc-Resources/stopersimg.webp",
        "../Doc-Resources/testimg.png",
        "../Doc-Resources/stopersimg.webp",
        "../Doc-Resources/stopersimg.webp",
        "../Doc-Resources/testimg.png",
        "../Doc-Resources/stopersimg.webp",
        "../Doc-Resources/stopersimg.webp",
    ]
    images = [Image.open(path) for path in image_paths]

    client = InferenceHTTPClient(
        api_url="http://localhost:9001",
        api_key="7dEiP3o3XQGNET8f4jlC" # Add your API key here
    )

    # Queues for images and detections
    image_queue = Queue()
    detection_queue = Queue()

    # Add images to the image queue
    for img in images:
        image_queue.put(img)

    # Create and start worker threads
    threads = [
        Thread(target=inference_worker, args=(image_queue, detection_queue, client), daemon=True),
        Thread(target=geomatics_worker, args=(detection_queue,), daemon=True)
    ]

    start = time.time()
    for thread in threads:
        thread.start()

    # Wait for queues to be processed
    image_queue.join()
    detection_queue.join()

    # Stop workers
    for _ in threads:
        image_queue.put(None)
        detection_queue.put(None)

    for thread in threads:
        thread.join()

    end = time.time()
    print(f"Total geomatics calculations performed: {global_counter}")
    print("Processing complete. Took {:.2f} seconds to complete".format(end - start))