# Third-party sources

The original project metadata declares MIT for this project's source. External libraries retain their own licenses.

The BFCL integration delegates scoring to the official Berkeley Function Calling Leaderboard implementation in https://github.com/ShishirPatil/gorilla . Commit pins and upstream URL are recorded in `src/exotic_trainer/bfcl.py`. No upstream evaluator tree or real benchmark dataset is bundled here. The adapter includes the public Liquid function-calling prompt; consult the relevant Liquid model card at https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct and the model's license before obtaining or redistributing weights.

Synthetic test cases are software fixtures, not conversations collected from users. The tiny example dataset was authored for this publication preparation. Historical research outcomes and private experiment exports are not included or claimed as independently reproduced results.
