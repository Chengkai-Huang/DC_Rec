# DCRec

## Reproduce the results

### Amazon Beauty Data

```
python -u main.py --data amazon_beauty  --timesteps 50  --mode train  --loss_type mix  --lr 0.004  --hidden_factor 128 --batch_size 512 --num_block 4  --dropout_rate 0.1
```
