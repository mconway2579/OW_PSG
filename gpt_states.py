from config import gpt_model, gpt_temp
import json
from PIL import Image
import base64
import io
import matplotlib.pyplot as plt
import numpy as np
import cv2
import re

def get_state_querry_prompt():
    """
    Returns a system prompt
    Parameters:
    - None
    Returns:
    - system prompt: str
    """
    system_prompt = ("""
You are a robots eyes. Your task is to analyze the scene, determine the objects present, infer their relationships through detailed chain of thought reasoning, and present a graph describing the scene.

# Instructions

You should output a scene graph with objects(nodes) and object relationships(edges) on their own lines with the following format:

- **object: str name of object.** a node in the graph representing an object in the scene.
  
- **object_relationship: a tuple like  <OBJECT1, RELATIONSHIP, OBJECT2>, where `OBJECT1` is related to `OBJECT2` via `Relationship`.** an edge in the graph
                     
# Chain of Thought Reasoning

1. **Identify Objects**: Begin by analyzing the scene to identify all visible objects.
    - List each object and the number of instances of that object.
  
2. **Determine Object Positions**: For each object, determine its placement in relation to other objects:
   - Is the object spatially related to another object?
   - Capture as many relationships as you can

3. **Establish Relationships**: Once object positions are determined, establish relationships following these rules:
   - Each relationship is a triple `<OBJECT1, RELATIONSHIP, OBJECT2>`, where `OBJECT1` is related to `OBJECT2` by RELATIONSHIP.

4. **Verify Completeness**: Ensure that all objects are covered in the relationships and that none remain unrelated.

# Output Graph Format

Your output should be formatted as with each object and object relationship on its own line, like the examples below:

object: A
object: B
object: C
object: Table
object_relationship: A, is on, B
object_relationship: B, is under, Table
object_relationship: C, is next to, B

object: Cat
object: Sofa
object: Coffee Table
object: Rug
object_relationship: Cat, is on, Sofa
object_relationship: Coffee Table, is next to, Sofa
object_relationship: Rug, is under, Coffee Table
object_relationship: Cat, is near, Rug                     

object: Phone
object: Desk
object: Chair
object: Window
object_relationship: Phone, is on, Desk
object_relationship: Chair, is next to, Desk
object_relationship: Desk, is under, Window
object_relationship: Chair, is in front of, Window

object: Book
object: Lamp
object: Plant
object: Shelf
object_relationship: Book, is on, Shelf
object_relationship: Lamp, is beside, Book
object_relationship: Plant, is under, Lamp
object_relationship: Plant, is next to, Shelf

object: Bicycle
object: Helmet
object: Backpack
object: Park Bench
object_relationship: Bicycle, is near, Park Bench
object_relationship: Helmet, is on, Bicycle
object_relationship: Backpack, is on, Bicycle
object_relationship: Park Bench, is behind, Bicycle

object: Guitar
object: Amplifier
object: Microphone
object: Stage
object_relationship: Guitar, is on, Stage
object_relationship: Amplifier, is next to, Guitar
object_relationship: Microphone, is in front of, Amplifier
object_relationship: Stage, is under, Microphone

object: Spaceship
object: Satellite
object: Planet
object: Asteroid
object_relationship: Spaceship, is near, Satellite
object_relationship: Satellite, is orbiting, Planet
object_relationship: Asteroid, is close to, Planet
object_relationship: Spaceship, is approaching, Asteroid

object: Cup
object: Saucer
object: Teapot
object: Table
object_relationship: Cup, is on, Saucer
object_relationship: Teapot, is next to, Cup
object_relationship: Saucer, is under, Teapot
object_relationship: Table, is behind, Teapot

                     
# Notes
- Ensure no object is left unplaced; every object must be included in the relationships field either on another object or on the table.
- Follow the reasoning steps explicitly before outputting to ensure correctness and completeness.
- You cannot have an object in a relationship but not in the object list or saftey will be at risk
- Ensure that the object_relationships are only made up of objects in the objects list
- Use specific Nouns and advjectives to label objects
- Include many relationships
# Requirements
- You must explicty reference each object in the relationships
- Each line must be tagged as an object (object: ) to be classified as a node
- Each line must be tagged as a relationship (object_relationship: ) to be classified as an edge
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
        response_format={"type": "text"},
        temperature=gpt_temp
    )
    
    output_text = state_response.choices[0].message.content
    print(f"\n\ngpt state {output_text=}")
    # Matches optional leading "- " and "**", then "object:" followed by text, ending with optional "**" and punctuation.
    object_pattern = re.compile(
        r'^\s*-?\s*\*{0,2}object:\s*(.+?)\*{0,2}\.?$', 
        re.MULTILINE
    )

    # Regex pattern for relationships:
    # Matches optional markdown markers, then "object_relationship:" followed by three comma-separated parts.
    relationship_pattern = re.compile(
        r'^\s*-?\s*\*{0,2}object_relationship:\s*(?P<obj1>.+?),\s*(?P<rel>.+?),\s*(?P<obj2>.+?)\*{0,2}\.?$', 
        re.MULTILINE
    )

    objects = object_pattern.findall(output_text)
    relationship_tuples = []

    for line in output_text.splitlines():
        if "object_relationship:" in line:
            match = relationship_pattern.match(line)
            if match:
                obj1 = match.group("obj1").strip()
                rel = match.group("rel").strip()
                obj2 = match.group("obj2").strip()
                relationship_tuples.append((obj1, rel, obj2))
            else:
                print(f"Failed to match relationship line: {line}")

    result = {
        "objects": objects,
        "object_relationships": relationship_tuples
    }
    print(f"\nparsed state {result=}\n\n")
    return state_response, result, state_querry_system_prompt, user_prompt

    
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
    state_response, state_json, state_querry_system_prompt, state_querry_user_prompt = get_state(client, rgb_img, "What do you see?", pose=pose)
    print_json(state_json)
    plt.figure()
    plt.imshow(rgb_img)
    plt.show(block = False)
    plt.pause(1)
    plt.show()

