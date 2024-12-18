from config import gpt_model, gpt_temp
import json
from PIL import Image
import base64
import io
import matplotlib.pyplot as plt
import numpy as np
import cv2


def get_state_querry_prompt():
    """
    Returns a system prompt
    Parameters:
    - None
    Returns:
    - system prompt: str
    """
    system_prompt = ("""
You are a robots eyes. Your task is to analyze the scene, determine the objects present, and infer their relationships through detailed chain of thought reasoning.

# Instructions

You should output a JSON object containing the following fields:

- **objects**: A list of strings where the string describes the object visible in the scene descriptions should be percise nouns.
  
- **object_relationships**: A list of tuples describing relationships between the objects. Each tuple should be in the format `<OBJECT1, RELATIONSHIP, OBJECT2>`, where `OBJECT1` is related to `OBJECT2`. 

# Chain of Thought Reasoning

1. **Identify Objects**: Begin by analyzing the scene to identify all visible objects.
    - List each object and the number of instances of that object.
  
2. **Determine Object Positions**: For each object, determine its placement in relation to other objects:
   - Is the object spatially related to another object?
   - Capture as many relationships as you can

3. **Establish Relationships**: Once object positions are determined, establish relationships following these rules:
   - Each relationship is a triple `<OBJECT1, RELATIONSHIP, OBJECT2>`, where `OBJECT1` is related to `OBJECT2` by RELATIONSHIP.

4. **Verify Completeness**: Ensure that all objects are covered in the relationships and that none remain unrelated.

# Output Format

Your output should be formatted as a JSON object, like the example below:

```json
{
  "objects": ["table", "A", "B", "C"],
  "object_relationships": [["A", "is on", "B"], ["B", "is under", "table"], ["C", "is next to", "B"]]
}

# Notes

- Ensure no object is left unplaced; every object must be included in the relationships field either on another object or on the table.
- Follow the reasoning steps explicitly before outputting to ensure correctness and completeness.
- You cannot have an object in a relationship but not in the object list or saftey will be at risk
- Ensure that the object_relationships are only made up of objects in the objects list
- Use specific Nouns and advjectives to label objects
- Include many relationships
""")
    return system_prompt


def rotate_image(image, yaw_degrees):
    """
    Rotates a numpy.ndarray (OpenCV image) by a specified angle.
    
    Parameters:
        image (numpy.ndarray): The input image.
        yaw_degrees (float): The angle in degrees to rotate (negative to counteract yaw).
    
    Returns:
        numpy.ndarray: The rotated image.
    """
    # Get image dimensions
    (h, w) = image.shape[:2]
    
    # Compute the center of the image
    center = (w // 2, h // 2)
    
    # Compute the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, -yaw_degrees, 1.0)
    
    # Perform the rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    
    return rotated_image

#helper function that formats the image for GPT api
def encode_image(img_array):
    """
    Encodes image as JPEG for use with OPENAI api
    """
    # Convert the ndarray to a PIL Image
    image = Image.fromarray(img_array)
    
    # Create a BytesIO object to save the image
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")  # Specify the format you want
    buffered.seek(0) #Possibly not needed
    # Get the byte data and encode to base64
    encoded_string = base64.b64encode(buffered.read()).decode('utf-8')
    
    return encoded_string

#api calling function
def get_state(client, rgb_image, user_prompt, pose=None):

    """
    Gets state json
    Parameters:
    - client: openai client
    - rgb_image: numpy array
    - user_prompt: str allows the user to provide more context

    Returns:
    - state_response: openai response object
    - state_json: json with entries for objects and object relationships
    - state_querry_system_prompt: str
    - user_prompt: str
    """
    if pose is not None:
        yaw = pose[5]
        yaw = np.degrees(yaw)
        #print(f"{yaw=}")
        rgb_image = rotate_image(rgb_image.copy(), yaw)
    

    encoded_img = encode_image(rgb_image)
    img_type = "image/jpeg"

    state_querry_system_prompt = get_state_querry_prompt()

    state_response = client.chat.completions.create(
        model=gpt_model,
        messages=[
            { "role": "system", "content":[{"type": "text", "text":f"{state_querry_system_prompt}"}]},  # Only text in the system message
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{img_type};base64,{encoded_img}"}}
                ]
            },
        ],
        response_format={"type": "json_object"},
        temperature=gpt_temp
    )
    
    state_json = json.loads(state_response.choices[0].message.content)
    return state_response, state_json, state_querry_system_prompt, user_prompt

    
def print_json(j, name=""):
    out_str = f"{name}={json.dumps(j, indent=4)}"
    print(out_str)
    return out_str


if __name__ == "__main__":
    from APIKeys import API_KEY
    from openai import OpenAI
    import pickle


    with open("./custom_dataset/one on two/top_view.pkl", "rb") as file:
        rgb_img, depth_img, pose, K, depth_scale = pickle.load(file)

    client = OpenAI(
        api_key= API_KEY,
    )
    state_response, state_json, state_querry_system_prompt, state_querry_user_prompt = get_state(client, rgb_img, "how are objects layed out on the table?", pose=pose)
    print_json(state_json)
    plt.figure()
    plt.imshow(rgb_img)
    plt.show(block = False)
    plt.pause(1)
    plt.show()

