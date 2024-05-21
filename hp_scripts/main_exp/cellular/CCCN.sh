# Description: Main experiment script for GCN model.
# ----Node regression datasets: US County Demographics----
task_variables=( 'Election' 'MedianIncome' 'MigraRate' 'BirthRate' 'DeathRate' 'BachelorRate' 'UnemploymentRate' )

for task_variable in ${task_variables[*]} 
do 
  python train.py \
    dataset=us_country_demos \
    dataset.parameters.data_seed=0,3,5,7,9 \
    dataset.parameters.task_variable=$task_variable \
    model=cell/cccn \
    model.feature_encoder.out_channels=32,64,128 \
    model.feature_encoder.proj_dropout=0.25,0.5 \
    model.backbone.n_layers=1,2,3,4 \
    model.optimizer.lr="0.01,0.001" \
    dataset.transforms.graph2cell_lifting.max_cell_length=10 \
    model.readout.readout_name="NoReadOut,PropagateSignalDown" \
    trainer.max_epochs=1000 \
    trainer.min_epochs=500 \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    logger.wandb.project=TopoBenchmarkX_Cellular \
    tags="[MainExperiment]" \
    --multirun
    
done

# ----Cocitation datasets----
datasets=( 'cocitation_cora' 'cocitation_citeseer' 'cocitation_pubmed' )

for dataset in ${datasets[*]}
do
  python train.py \
    dataset=$dataset \
    dataset.parameters.data_seed=0,3,5,7,9 \
    model=cell/cccn \
    model.feature_encoder.out_channels=32,64,128 \
    model.feature_encoder.proj_dropout=0.25,0.5 \
    model.backbone.n_layers=1,2 \
    model.optimizer.lr="0.01,0.001" \
    dataset.transforms.graph2cell_lifting.max_cell_length=10 \
    model.readout.readout_name="NoReadOut,PropagateSignalDown" \
    trainer.max_epochs=500 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=25 \
    logger.wandb.project=TopoBenchmarkX_Cellular \
    tags="[MainExperiment]" \
    --multirun
done

# # ----Graph regression dataset----
# # Train on ZINC dataset
# python train.py \
#   dataset=ZINC \
#   seed=42,3,5,23,150 \
#   model=cell/cccn \
#   model.optimizer.lr=0.01,0.001 \
#   model.optimizer.weight_decay=0 \
#   model.feature_encoder.out_channels=32,64,128 \
#   model.backbone.n_layers=2,4 \
#   model.feature_encoder.proj_dropout=0.25,0.5 \
#   dataset.parameters.batch_size=128,256 \
#   dataset.transforms.one_hot_node_degree_features.degrees_fields=x \
#   dataset.parameters.data_seed=0 \
#   dataset.transforms.graph2cell_lifting.max_cell_length=10 \
#   model.readout.readout_name="NoReadOut,PropagateSignalDown" \
#   logger.wandb.project=TopoBenchmarkX_Cellular \
#   trainer.max_epochs=500 \
#   trainer.min_epochs=50 \
#   callbacks.early_stopping.min_delta=0.005 \
#   trainer.check_val_every_n_epoch=5 \
#   callbacks.early_stopping.patience=10 \
#   tags="[MainExperiment]" \
#   --multirun

# ----TU graph datasets----
# MUTAG have very few samples, so we use a smaller batch size
# Train on MUTAG dataset
python train.py \
  dataset=MUTAG \
  model=cell/cccn \
  model.optimizer.lr=0.01,0.001 \
  model.feature_encoder.out_channels=32,64,128 \
  model.backbone.n_layers=1,2,3,4 \
  model.feature_encoder.proj_dropout=0.25,0.5 \
  dataset.parameters.data_seed=0,3,5,7,9 \
  dataset.parameters.batch_size=32,64 \
  dataset.transforms.graph2cell_lifting.max_cell_length=10 \
  model.readout.readout_name="NoReadOut,PropagateSignalDown" \
  trainer.max_epochs=500 \
  trainer.min_epochs=50 \
  trainer.check_val_every_n_epoch=1 \
  logger.wandb.project=TopoBenchmarkX_Cellular \
  callbacks.early_stopping.patience=25 \
  tags="[MainExperiment]" \
  --multirun

# Train rest of the TU graph datasets
datasets=( 'PROTEINS_TU' 'NCI1' 'NCI109' 'IMDB-BINARY' 'IMDB-MULTI' )

for dataset in ${datasets[*]}
do
  python train.py \
    dataset=$dataset \
    model=cell/cccn \
    model.optimizer.lr=0.01,0.001 \
    model.feature_encoder.out_channels=32,64,128 \
    model.backbone.n_layers=1,2,3,4 \
    model.feature_encoder.proj_dropout=0.25,0.5 \
    dataset.parameters.data_seed=0,3,5,7,9 \
    dataset.parameters.batch_size=128,256 \
    dataset.transforms.graph2cell_lifting.max_cell_length=10 \
    model.readout.readout_name="NoReadOut,PropagateSignalDown" \
    logger.wandb.project=TopoBenchmarkX_Cellular \
    trainer.max_epochs=500 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=5 \
    callbacks.early_stopping.patience=10 \
    --multirun
done

# ----Heterophilic datasets----

datasets=( roman_empire minesweeper )

for dataset in ${datasets[*]}
do
  python train.py \
    dataset=$dataset \
    model=cell/cccn \
    model.optimizer.lr=0.01,0.001 \
    model.feature_encoder.out_channels=32,64,128 \
    model.backbone.n_layers=1,2,3,4 \
    model.feature_encoder.proj_dropout=0.25,0.5 \
    dataset.parameters.data_seed=0,3,5,7,9 \
    dataset.parameters.batch_size=1 \
    dataset.transforms.graph2cell_lifting.max_cell_length=10 \
    model.readout.readout_name="NoReadOut,PropagateSignalDown" \
    logger.wandb.project=TopoBenchmarkX_Cellular \
    trainer.max_epochs=1000 \
    trainer.min_epochs=50 \
    trainer.check_val_every_n_epoch=1 \
    callbacks.early_stopping.patience=50 \
    tags="[MainExperiment]" \
    --multirun
done