# T2IScoreScore

### **Who Evaluates the Evaluations? Assessing the Faithfulness and Consistency of Text-to-Image Evaluation Metrics with *T2IScoreScore***


We introduced **T2IScoreScore** dataset! 📸✨  🤗 [HF Repo](https://huggingface.co/datasets/saxon/T2IScoreScore) 

T2IScoreScore is a meticulously curated dataset featuring image sets that smoothly transition from high to low faithfulness with respect to the given prompt. This collection includes both synthetic and natural examples, providing a comprehensive range for meta-evaluating existing Text-to-Image (T2I) metrics. Our goal is to facilitate the development of more accurate and consistent T2I metrics in the future.


⚡	There is a sample of an error graph for the initial prompt “**A
happy young woman eating a tasty orange donut**” using natural
images. There are two different nodes of error distance 1 from the
head node of the graph:



<img src="figures/Sample.png" style="width:450px; height:300px;">


### Tested Scores on This Dataset

- **ClipScore**
- **BlipScore**
- **AlignScore**

We created questions with [DSG](https://github.com/j-min/DSG), [TIFA](https://github.com/Yushi-Hu/tifa) and tested with the performance of these VQA models:

- **LLaVa**
- **Fuyu**
- **mPLUG**
