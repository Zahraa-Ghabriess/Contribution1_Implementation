import onnxruntime
from optimum.onnxruntime import ORTModelForFeatureExtraction
from optimum.onnxruntime import ORTOptimizer, AutoOptimizationConfig


# Function to create, optimize and quantize BERT model using ONNX Runtime
# By default TinyBERT Base Model used for feature extraction

def create_bert_model(save_directory = "BERT_Model/", base_model_id = "huawei-noah/TinyBERT_General_4L_312D"):
    session_options = onnxruntime.SessionOptions()
    session_options.log_severity_level = 0

    # TinyBERT Base Model used for feature extraction
    base_model = ORTModelForFeatureExtraction.from_pretrained(base_model_id,
                                                            export=True,
                                                            provider="CUDAExecutionProvider",
                                                            #session_options=session_options   #to check whether all nodes are indeed placed on the CUDA execution provider or not
                                                            )
    base_model.save_pretrained(save_directory)


    # Apply graph optimization
    print("Performing Graph Opitimization...")

    # Step 1: Define the optimization methodology
    # Here the optimization level is selected to be 2, enabling basic optimizations
    # such as redundant node eliminations and constant folding and some extended optimizations. Higher optimization
    # level will result in a hardware dependent optimized graph.

    optimization_config = AutoOptimizationConfig.O2() # basic and extended general optimizations, transformers-specific fusions.
    optimizer = ORTOptimizer.from_pretrained(base_model)

    # Step 2: Opitimize the model
    optimizer.optimize(save_dir=save_directory, optimization_config=optimization_config)


    # Load the final model
    base_model = ORTModelForFeatureExtraction.from_pretrained(
                                                                save_directory, 
                                                                file_name = "model_optimized.onnx", 
                                                                provider="CUDAExecutionProvider"
                                                            )

    return base_model