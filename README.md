# Super-Segger-Toolkit
Toolkit for linking cells from masks matrix, providing the empty graph to be filled in, and visualization tools like lineage, phase image.

The main idea is to store the linking result as a Graph of Cells:

- **Class Cell:**
  - `self.frame`
  - `self.label`
  - `self.polygon`
- **Class Graph:**
  - `self.node`
  - `self.edge: (node, node)`

---
The main data **resource manager, called LinkComposer**, stores phase/mask directories and cells set. It helps in quickly plotting information.

## Steps:
1. Create a LinkComposer:
   - `LinkComposer.read_folder(mask_folder, phase_folder)`
   - `LinkComposer.read_tif(mask_tif, phase_tif)`
2. Ask composer for an empty directed graph: `composer.make_new_directed_graph()`
3. Fill in edges on the tree: `composer.link(G, cell1, cell2)`
4. Check results by visualization:
     - `visualizer.quick_lineage(G)`
     - `visualizer.get_label_info(G), composer.show_frame_phase(...)`
5. Finished. Also, multiple linker comparisons can be done, if desired.

   
Read Demo for more information: 

---
## Visualization:
**Phase tracking videos:**
Cell Event Tracking Labeling:

![Image 1](https://github.com/yyang35/super-segger-toolkit/blob/main/readme_media/event_label.gif)

Ancient/generation Labeling:
![Image 2](https://github.com/yyang35/super-segger-toolkit/blob/main/readme_media/generation_label.gif)

Warning Error Labeling:
![Image 3](https://github.com/yyang35/super-segger-toolkit/blob/main/readme_media/warning_label.gif)

**Lineage comparsion:**
![image](https://github.com/yyang35/super-segger-toolkit/blob/main/readme_media/lineage_comparsion.png)

**Cross time phase video compasion:**
![image](https://github.com/yyang35/super-segger-toolkit/blob/main/readme_media/phase_comparsion.png)
