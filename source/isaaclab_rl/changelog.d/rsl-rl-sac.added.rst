* Added typed SAC model, distribution, network, and algorithm configurations with native RSL-RL defaults.
* Forwarded terminal observations through the vector wrapper without changing the environment's extras.
* Added the SAC ``grad_reduce_bucket_mb`` and ``validate_final_obs`` options for multi-GPU training and final-observation validation.
