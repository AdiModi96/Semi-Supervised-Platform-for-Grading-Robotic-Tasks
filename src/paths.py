import inspect
import os

current_file_path = inspect.getfile(inspect.currentframe())

project_folder_path = os.path.abspath(os.path.join(current_file_path, '..', '..'))
src_folder_path = os.path.join(project_folder_path, 'src')
data_folder_path = os.path.join(project_folder_path, 'data')
trained_models_weights_folder_path = os.path.join(project_folder_path, 'trained model\'s weights')
auto_annotation_videos_folder_path = os.path.join(project_folder_path, 'auto annotated')

sb_data_folder_path = os.path.join(data_folder_path, 'SB')
