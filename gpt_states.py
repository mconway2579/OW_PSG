from config import gpt_model, gpt_temp
import json
from PIL import Image
import base64
import io

def get_state_querry_prompt():
    system_prompt = ("""
You are a robots eyes. Your task is to analyze the scene, determine the objects present, and infer their relationships through detailed chain of thought reasoning.

# Instructions

You should output a JSON object containing the following fields:

- **objects**: A list of all objects visible in the scene.
  
- **object_relationships**: A list of tuples describing relationships between the objects. Each tuple should be in the format `<OBJECT1, RELATIONSHIP, OBJECT2>`, where `OBJECT1` is related to `OBJECT2`. 

# Chain of Thought Reasoning

1. **Identify Objects**: Begin by analyzing the scene to identify all visible objects.
  
2. **Determine Object Positions**: For each object, determine its placement in relation to other objects:
   - Is the object on another block or on the table?
   - Make sure no object is left unplaced.

3. **Establish Relationships**: Once object positions are determined, establish relationships following these rules:
   - Record relationships where one object is directly on top of another.
   - Each relationship is a triple `<OBJECT1, RELATIONSHIP, OBJECT2>`, where `OBJECT1` is related to `OBJECT2`.

4. **Verify Completeness**: Ensure that all objects are covered in the relationships and that none remain without being stacked or placed on the table.

# Output Format

Your output should be formatted as a JSON object, like the example below:

```json
{
  "objects": ["table", "A", "B", "C"],
  "object_relationships": [["A", "is on", "B"], ["B", "is on", "table"], ["C", "is on", "table"], ["C", "is next to", "B"]]
}
```

Make sure the output JSON adheres strictly to the specified structure and validates that each object is accounted for in the relationships.

# Examples

**Input Scene Description**:
- A is on B.
- B is on the table.
- C is also on the table.
- C is next to B.

**Chain of Thought Reasoning**:
1. Identify Objects: The scene includes "A", "B", "C", and the "table".
2. Determine Object Positions:
   - A is on B.
   - B is on the table.
   - C is on the table.
   - C is next to B.
3. Establish Relationships:
   - `<A, is on, B>`
   - `<B, is on, Table>`
   - `<C, is on, Table>`
   - `<C, is next to, B>`
                     

**Output JSON**:
```json
{
  "objects": ["table", "A", "B", "C"],
  "object_relationships": [["A", "is on", "B"], ["B", "is on", "table"], ["C", "is on", "table"], ["C", "is next to", "B"]]
}
```

# Notes

- Ensure no object is left unplaced; every object must be included in the relationships field either on another object or on the table.
- Follow the reasoning steps explicitly before outputting to ensure correctness and completeness.
- You cannot have an object in a relationship but not in the object list or saftey will be at risk
- Ensure that the object_relationships are only made up of objects in the objects list
""")
    user_prompt = f"Give me the state in the given image"
    return system_prompt, user_prompt

#helper function that formats the image for GPT api
def encode_image(img_array):
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
def get_state(client, rgb_image):
    image = encode_image(rgb_image)
    img_type = "image/jpeg"

    state_querry_system_prompt, state_querry_user_prompt = get_state_querry_prompt()

    #print(f"{state_querry_system_prompt=}")
    #print()
    #print(f"{state_querry_user_prompt=}")
    #print()
    state_response = client.chat.completions.create(
        model=gpt_model,
        messages=[
            { "role": "system", "content":[{"type": "text", "text":f"{state_querry_system_prompt}"}]},  # Only text in the system message
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": state_querry_user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{img_type};base64,{encode_image(rgb_image)}"}}
                ]
            },
        ],
        response_format={"type": "json_object"},
        temperature=gpt_temp
    )
    state_json = json.loads(state_response.choices[0].message.content)
    return state_response, state_json, state_querry_system_prompt, state_querry_user_prompt

    
def print_json(j, name=""):
    out_str = f"{name}={json.dumps(j, indent=4)}"
    print(out_str)
    return out_str


if __name__ == "__main__":
    from APIKeys import API_KEY
    import matplotlib.pyplot as plt
    from openai import OpenAI
    import numpy as np
    import cv2
    import os

    sample = "NYU1449"
    parent_path = f"./SUNRGBD/kv1/NYUdata/{sample}/"
    rgb_path = os.path.join(parent_path, f"image/{sample}.jpg")
    depth_path = os.path.join(parent_path, f"depth/{sample}.png")

    rgb_image = cv2.imread(rgb_path)
    depth_image = cv2.imread(depth_path)

    

    client = OpenAI(
        api_key= API_KEY,
    )
    state_response, state_json, state_querry_system_prompt, state_querry_user_prompt = get_state(client, rgb_image)
    print_json(state_json)
    
        
    fig, axes = plt.subplots(ncols=2)
    axes[0].imshow(rgb_image)
    axes[1].imshow(depth_image)

    plt.show()

