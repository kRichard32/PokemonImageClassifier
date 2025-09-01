import json
import os
from pathlib import Path

import psycopg2

import tritonclient.http as httpclient
from PIL import Image
from torchvision import transforms
import argparse

def preprocess(img):
    transformed_img = transforms.Resize((220, 220))(transforms.ToTensor()(img)).unsqueeze(0).numpy()
    return transformed_img
def fix_perms_windows(abs_path):
    import win32security
    import ntsecuritycon as con
    userx, domain, type = win32security.LookupAccountName("", "Everyone")
    sd = win32security.GetFileSecurity(abs_path, win32security.DACL_SECURITY_INFORMATION)
    dacl = sd.GetSecurityDescriptorDacl()  # instead of dacl = win32security.ACL()
    dacl.AddAccessAllowedAce(win32security.ACL_REVISION, con.FILE_GENERIC_READ, userx)
    dacl.AddAccessAllowedAce(win32security.ACL_REVISION, con.FILE_GENERIC_WRITE, userx)
    sd.SetSecurityDescriptorDacl(1, dacl, 0)
    win32security.SetFileSecurity(abs_path, win32security.DACL_SECURITY_INFORMATION, sd)

def run_classification(image_path=None, identity=None):
    if image_path:
        img = Image.open(image_path).convert('RGB')
    else:
        connection = psycopg2.connect(database="postgres", user="postgres", password="1", host="localhost", port="5432")
        cursor = connection.cursor()
        cursor.execute("SELECT image from image_file where id=%s", (identity,))
        record = cursor.fetchone()[0]
        script_path = os.path.dirname(os.path.abspath(__file__))
        abs_path = os.path.join(script_path, "temp_img_data.png")

        file = Path(abs_path)
        file.touch(exist_ok=True)
        os.chmod(abs_path, 0o777)
        if os.name == 'nt':
            fix_perms_windows(abs_path)
        cursor.execute("SELECT lo_export(%s, %s)", (record, abs_path))
        img = Image.open(abs_path).convert('RGB')

    img = preprocess(img)
    client = httpclient.InferenceServerClient("localhost:8003")

    inputs = httpclient.InferInput("x", img.shape, "FP32")

    inputs.set_data_from_numpy(img, binary_data=True)

    outputs = httpclient.InferRequestedOutput("log_softmax", binary_data=True, class_count=1000)
    results = client.infer(model_name="pokemon_prediction_model", inputs=[inputs], outputs=[outputs])
    inference_output = results.as_numpy("log_softmax").astype(str)
    max_element = int(inference_output[0].split(":")[1])
    script_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_path, "TrainingAndClient/global_list.json"), "r") as f:
        global_label_set = json.load(f)
    print(global_label_set[max_element])
    return global_label_set[max_element]
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-im",
        "--image-path",
        default=False,
        help="Path to the image",
    )
    parser.add_argument(
        "-id",
        "--id",
        default=False,
        help="Id of image in database",
    )
    FLAGS = parser.parse_args()
    run_classification(FLAGS.image_path, FLAGS.id)


