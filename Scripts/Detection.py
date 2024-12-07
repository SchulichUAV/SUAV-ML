'''
Final combined detection file.
'''
import time
import base64
import cv2 
import numpy as np
from inference_sdk import InferenceHTTPClient
from PIL import Image

def detect_objects(images: list[Image.Image], client: InferenceHTTPClient) -> None:
    """
    Detects objects in a list of images using the detection workflow

    Parameters:
    images (List[PIL.Image.Image]): List of PIL images to be processed.
    client (InferenceHTTPClient): Inference client to run the workflow.
    """
    try:
        for img in images:
            # Convert PIL Image to OpenCV format
            cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            resized_image = cv2.resize(cv_image, (640, 640), interpolation=cv2.INTER_AREA)

            # Convert the resized image to base64 format
            _, buffer = cv2.imencode('.png', resized_image)
            base64_image = base64.b64encode(buffer).decode('utf-8')

            # Run the inference workflow
            start_time = time.time()
            result = client.run_workflow(
                workspace_name="suavcoco",
                workflow_id="combined-models",
                images={
                    "image": base64_image 
                }
            )
            
            end_time = time.time()
            print(f"Workflow execution time: {end_time - start_time:.2f} seconds")

            if not result or "output" not in result[0]:
                raise ValueError("Invalid workflow result. Missing 'output' key.")

            # Decode the result image
            image_string = result[0]["output"]
            image_data = base64.b64decode(image_string)
            nparr = np.frombuffer(image_data, np.uint8)
            output_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            cv2.imshow("Object Detection", output_image)

            key = cv2.waitKey(0)
            if key == ord('q'):
                print("Quitting...")
                break

        cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # ---- Add Images Here ----
    image_paths = ["", ""]
    images = [Image.open(path) for path in image_paths]

    client = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key=""  # ---- Add your API key here ----
    )

    detect_objects(images, client)
