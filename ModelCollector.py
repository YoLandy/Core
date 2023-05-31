import os

# Models import
model_names = []
for root, dirs, files in os.walk("models"):  
    for filename in files:
        model_name = filename.split('.')[0]
        ftype = filename.split('.')[-1]
        if ftype == 'py' and model_name[0] != "_":
            exec(f'from models.{model_name} import {model_name}')
            model_names.append(model_name)

# Models inits and descriptions
models = {model_name: eval(f'{model_name}()') for model_name in model_names}  
model_descriptions = {model_name: model.description for model_name, model in models.items()}

# Models info
models_info = {
    model_name: {
        'description': model.description,
        'inputs': model.input_type,
        'outputs': model.output_type
    } for model_name, model in models.items()
} 

if __name__ == '__main__':
    print(models_info)