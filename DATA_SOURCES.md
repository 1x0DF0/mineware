# Minecraft AI training data — map of the internet

You cannot download “all Minecraft play,” but these are the real, usable public sources.

## A. Screen-pixel object detection (what mineware uses now)

| Source | What | Size | How we use it |
|--------|------|------|----------------|
| **Your sessions** (`sessions/*/frames`) | Your 26.2 play | hundreds of frames | Auto-labeled trees → YOLO |
| **dataset_raw/** | collect_data dumps | small | Same |
| **HuggingFace mobs YOLO** [`hmnshudhmn24/minecraft-mobs-yolo-dataset`](https://huggingface.co/datasets/hmnshudhmn24/minecraft-mobs-yolo-dataset) | MC screenshots + boxes | ~2.5k images | Multi-class: creeper/skeleton/… (class id ≥1) |
| **Kaggle** [Minecraft Mobs YOLO](https://www.kaggle.com/datasets/dracotlw/minecraft-mobs-yolo-dataset) | Same family | ~2.5k | Mirror of above |
| **Roboflow Universe** [search minecraft](https://universe.roboflow.com/search?q=class%3Aminecraft) | Tree / mob / player sets | varies | Needs free API key → `fetch_public_dataset.py` |
| Roboflow: [minecraft-tree-detection](https://universe.roboflow.com/minecraft-thing/minecraft-tree-detection) | Trees | ~200–300 | Best public **tree** boxes |
| Roboflow: [tree-wood ID](https://universe.roboflow.com/ananthv/minecraft-tree-wood-identification-dataset) | Trunks | small | Game-automation blog set |
| Roboflow: [minecraft-yolo](https://universe.roboflow.com/minecraft-test/minecraft-yolo-v1xlv) | Mixed | ~1.7k | Export YOLOv8 |
| Roboflow: [mob-detection](https://universe.roboflow.com/minecraft-object-detection/minecraft-mob-detection) | Mobs | ~3.6k | Combat later |

**Command we run:**
```powershell
py .\build_training_corpus.py   # local frames + any external_datasets/
py .\train_yolo.py --epochs 50 --name run2
# optional Roboflow trees:
$env:ROBOFLOW_API_KEY="..."
py .\fetch_public_dataset.py
py .\train_yolo.py --epochs 50 --name run3
```

## B. Full agent / action data (not YOLO boxes)

| Source | What | Notes for mineware |
|--------|------|--------------------|
| **OpenAI VPT** [github.com/openai/Video-Pre-Training](https://github.com/openai/Video-Pre-Training) | Huge contractor demos + IDM + BC weights | **~TB scale**. Action labels for mouse/keyboard in MC. Not drop-in for our YOLO+FSM; future BC/VPT branch. |
| **MineStudio** [CraftJarvis/MineStudio](https://github.com/CraftJarvis/MineStudio) | VPT data on HF, training stack | Same — trajectory policy, not screen boxes |
| **MineRL / BASALT** | Env + demos | Needs Malmo-style env, not Java 26.2 window grab |
| **Your `record_session.py`** | state+action jsonl | **Primary path for *our* special AI** on 26.2 |

VPT/MineRL teach *policies in their env*. We teach *perception on your pixels* + *policy on your inputs*. Both are valid; they don’t fully replace each other.

## C. What “go through the entire internet” realistically yields

1. **~thousands** of labeled MC screenshots (mobs, some trees) — free, HF/Roboflow  
2. **Your** hundreds of 26.2 frames — most important for GUI scale / version  
3. **VPT-scale** video+actions — optional later, heavy  
4. No free perfect “chop tree bounding boxes for 26.2” pack  

## D. Recommended data diet for *this* project

1. Keep recording play (`record_session.py`) — trees, ores, combat  
2. `build_training_corpus.py` after each session  
3. Roboflow tree set when you have an API key  
4. HF mobs set for multi-object vision  
5. Retrain YOLO → point agent at `minecraft_yolo/run2/weights/best.pt`  
6. Retrain BC (`train_policy.py`) when you have 30+ min of clean key logs  

## E. Related open projects (ideas / code, not auto-merged)

- OpenAI VPT — foundation Minecraft agent research  
- MineStudio / CraftJarvis — modern packaging of VPT-style agents  
- Roboflow game-automation blog — tree trunk detection tutorial  
- Various student YOLO Minecraft mob detectors on GitHub/HF  
