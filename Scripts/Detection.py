
import os
import itertools
import time
import base64
import cv2
import numpy as np
from PIL import Image
from inference_sdk import InferenceHTTPClient
from threading import Thread
from queue import Queue

global_counter = 0
scanned_index = 0

def run_inference_batch(base64_images, client):
    """Runs inference on a batch of images using the detection workflow."""
    try:
        print(f"Workflow started for batch of {len(base64_images)} images")

        start_time = time.time()
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

def geomatics_calculation(detections):
    """Performs geomatics calculations."""
    global global_counter
    global_counter += 1
    try:
        print("Performing geomatics calculations...")
        time.sleep(0.4)  # Simulating processing time
        print(f"Geomatics calculations complete")
    except Exception as e:
        print(f"Geomatics calculation error: {e}")

def scan_directory_for_images(image_queue, image_folder="../images", batch_size=12):
    """
    Scans the directory for new images, adds them to the queue in batches of 12,
    and keeps track of processed files.
    """
    # global scanned_index
    global scanned_index
    new_images = []
    # Get all image files
    for file in itertools.islice(os.listdir(image_folder), scanned_index, None):  # Sort to process in order
        print("scanned_image", file)
        file_path = os.path.join(image_folder, file)
        new_images.append(file_path)
        
        # Track scanned images
        scanned_index+=1

        if len(new_images) == batch_size:
            break  # Stop when we collect 12 images

    if not new_images:
        return False  # No new images found

    for img_path in new_images:
        image_queue.put(img_path)

    return True

def inference_worker(image_queue, detection_queue, client):
    """Worker thread to process images in batches and run inference."""
    while True:
        batch = []
        
        # Collect images in batch
        while len(batch) < 12 and not image_queue.empty():
            img_path = image_queue.get()
            if img_path is None:
                return

            try:
                # Load image
                pil_image = Image.open(img_path)
                cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                
                # Convert to base64
                _, buffer = cv2.imencode('.png', cv_image)
                base64_image = base64.b64encode(buffer).decode('utf-8')

                batch.append(base64_image)

            except Exception as e:
                print(f"Error processing image {img_path}: {e}")

            image_queue.task_done()

        if not batch:
            time.sleep(1)
            continue  # No images to process

        results = run_inference_batch(batch, client)

        if results:
            for result in results:
                for detection in result[0]['consensus_predictions']['predictions']:
                    detection_queue.put(detection)

def geomatics_worker(detection_queue):
    """Worker thread to process detections and perform geomatics calculations."""
    while True:
        detections = detection_queue.get()
        if detections is None:
            return
        geomatics_calculation(detections)
        detection_queue.task_done()

if __name__ == "__main__":
    client = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="7dEiP3o3XQGNET8f4jlC"  # Add your API key
    )

    image_queue = Queue()
    detection_queue = Queue()

    threads = [
        Thread(target=inference_worker, args=(image_queue, detection_queue, client), daemon=True),
        Thread(target=geomatics_worker, args=(detection_queue,), daemon=True)
    ]

    for thread in threads:
        thread.start()

    # Continuously scan for new images
    while True:
        has_new_images = scan_directory_for_images(image_queue)
        if not has_new_images:
            print("No new images found. Waiting...")
            time.sleep(0.5)  # Wait before rescanning

    # Stop threads properly (not reached in infinite loop)
    for _ in threads:
        image_queue.put(None)
        detection_queue.put(None)

    for thread in threads:
        thread.join()
