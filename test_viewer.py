import mujoco
import mujoco.viewer

# scene.xml dosyanı doğrudan yükleyip interaktif arayüzü başlatıyoruz
model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)
mujoco.viewer.launch(model, data)