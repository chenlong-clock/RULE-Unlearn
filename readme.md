# RULE: Reinforcement UnLEarning

This repository provides a framework for Reinforcement UnLEarning (RULE), based on the [EasyR1](https://github.com/hiyouga/EasyR1).

## Preprint
The preprint for this work is available at [arXiv:2506.07171](https://arxiv.org/abs/2506.07171).


## Directory Structure

```
log/            # Training and evaluation logs
# For Rejection Steering:
RS/             # Rejection Steering (RS) implementation
    scripts/    # Scripts for running RS experiments
    models/     # RS model implementations
    utils/      # Utility functions for RS
# For Refusal Boundary Optimization:
examples/       # Example experiment configs
data/           # Datasets and metadata
verl/           # Core source code (models, training, evaluation, utils)
run_muse.sh     # Script to run MUSE experiments for ReBO
run_rwku.sh     # Script to run RWKU experiments for ReBO
requirements.txt
setup.py
```

**Install the package**
   ```sh
   python setup.py install
   ```
   or 
   ```sh
   pip install -e .
   ```

## Run experiments


**Step 1. Run Rejection Steering**
   ```sh
   cd RS && bash scripts/full/run_rt_epoch_target.sh
   ```
**Step 2. Run Refusal Boundary Optimization**
   ```sh
   bash examples/exp_target/RWKU/run_llama_bs32_kl1e-2_forget_bf16_two_stage_reject_ref_rollout8_withformat_with_fb_neighbor_abs_lr2e-6.sh
   ```

## Configuration

- For RS:
Edit the `RS/scripts/full/run_rt_epoch_target.sh` script.

- For ReBO:
Edit YAML files in `examples/` to set hyperparameters, dataset paths, and model options.

## Acknowledgements

This project is built on top of EasyR1[https://github.com/hiyouga/EasyR1] and RWKU[https://github.com/jinzhuoran/RWKU]

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
## Cite our work
If you use this code in your research, please cite our preprint:

```bibtex
@misc{zhang2025rulereinforcementunlearningachieves,
      title={RULE: Reinforcement UnLEarning Achieves Forget-Retain Pareto Optimality}, 
      author={Chenlong Zhang and Zhuoran Jin and Hongbang Yuan and Jiaheng Wei and Tong Zhou and Kang Liu and Jun Zhao and Yubo Chen},
      year={2025},
      eprint={2506.07171},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.07171}, 
}
```