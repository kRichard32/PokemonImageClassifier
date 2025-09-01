import triton_python_backend_utils as pb_utils
# from ImageClassifier import run_inference_on_image, get_pokemon_label
from torchvision import transforms
class TritonPythonModel:
    # def initialize(self, args):
    #
    def execute(self, requests):
        responses = []
        # for request in requests:
        #     input_tensor = pb_utils.get_input_tensor_by_name(request, "input")
        #     answer = get_pokemon_label(run_inference_on_image(input_tensor))
        #     responses.append(answer)

        return responses