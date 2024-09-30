import os
import cv2
import torch
from scipy.io import savemat, loadmat


class FileOperations:
    def __init__(self, device, save_path, source_path, audio_path):
        self.device = device
        self.save_path = save_path
        self.source_name = source_path.split("/")[-1].split(".")[0]
        self.audio_name = audio_path.split("/")[-1].split(".")[0]

        self.source_folder_path = self.create_folders_if_not_exist()

        if os.path.exists(os.path.join(self.source_folder_path, "preprocessed_inputs")):
            self.preprocessed_inputs_exist = True
        else:
            self.preprocessed_inputs_exist = False

    def save(self, batch):
        self.save_output(rendered_frame_list=batch["rendered_frame_list"],
                         num_frames=batch["num_frames"],
                         time=batch["time"],
                         audio_path=batch["audio_path"],
                         original_frame=batch["original_frame"],
                         face_crop_coords=batch["face_crop_coords"])
        
        if not self.preprocessed_inputs_exist:
            self.save_inputs(source_type=batch["source_type"],
                            rendering_input_face=batch["rendering_input_face"],
                            face_crop_coords=batch["face_crop_coords"],
                            original_frame=batch["original_frame"],
                            source_coeff=batch["source_coeff"])
        
    def save_inputs(self, source_type, rendering_input_face, face_crop_coords, original_frame, source_coeff):
        input_folder_path = os.path.join(self.source_folder_path, "preprocessed_inputs")
        os.makedirs(input_folder_path, exist_ok=True)

        batch_to_save = {"source_type": source_type,
                         "rendering_input_face": rendering_input_face.detach().cpu().numpy(),
                         "face_crop_coords": face_crop_coords,
                         "original_frame": original_frame,
                         "source_coeff": source_coeff.detach().cpu().numpy()}
        
        savemat(os.path.join(input_folder_path, "batch.mat"), batch_to_save)

    def load_inputs(self):
        batch_inputs = loadmat(os.path.join(self.source_folder_path, "preprocessed_inputs", "batch.mat"))
        batch_to_load = {"source_type": batch_inputs["source_type"][0],
                         "rendering_input_face": torch.tensor(batch_inputs["rendering_input_face"]).to(self.device),
                         "face_crop_coords": list(batch_inputs["face_crop_coords"][0]),
                         "original_frame": batch_inputs["original_frame"],
                         "source_coeff": torch.tensor(batch_inputs["source_coeff"]).to(self.device)}
        return batch_to_load

    def save_output(self, rendered_frame_list, num_frames, time, audio_path, original_frame, face_crop_coords):
        tmp_folder_path = os.path.join(self.source_folder_path, "tmp")
        os.makedirs(tmp_folder_path)

        video_name = f"{self.audio_name}_{time}.mp4"
        
        original_frame = original_frame
        face_crop_coords = face_crop_coords
        original_width = face_crop_coords[2]-face_crop_coords[0]
        original_height = face_crop_coords[3]-face_crop_coords[1]
        for idx, rendered_frame in enumerate(rendered_frame_list):
            rendered_frame = (rendered_frame[0].permute(1,2,0).detach().cpu().numpy()*255).astype("uint8")
            original_frame[face_crop_coords[1]:face_crop_coords[3], face_crop_coords[0]:face_crop_coords[2]] = cv2.resize(rendered_frame, (original_width, original_height))
            cv2.imwrite(f"{tmp_folder_path}/{str(idx).zfill(len(str(num_frames)))}.png",
                        cv2.cvtColor(original_frame, cv2.COLOR_RGB2BGR))

        os.system(f"ffmpeg -y -hide_banner -loglevel error -framerate 25 -pattern_type glob -i '{tmp_folder_path}/*.png' -c:v libx264 -pix_fmt yuv420p {os.path.join(self.source_folder_path, video_name.replace('.mp4', '_novoice.mp4'))}")
        os.system(f"rm -rf {tmp_folder_path}")
        os.system(f"ffmpeg -hide_banner -loglevel error -i {os.path.join(self.source_folder_path, video_name.replace('.mp4', '_novoice.mp4'))} -i {audio_path} -map 0:v -map 1:a -c:v copy -shortest {os.path.join(self.source_folder_path, video_name)}")
        os.system(f"rm -rf {os.path.join(self.source_folder_path, video_name.replace('.mp4', '_novoice.mp4'))}")

    def create_folders_if_not_exist(self):
        os.makedirs(self.save_path, exist_ok=True)

        source_folder_path = os.path.join(self.save_path, self.source_name)
        os.makedirs(source_folder_path, exist_ok=True)
        return source_folder_path