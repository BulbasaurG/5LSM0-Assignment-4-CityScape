import time

def log_train_val(df_train,df_val,csv_name, model_root_dir):
  log_time = time.strftime('%d-%m-%Y-%H-%M-%S',time.localtime())
  train_csv_path = f'{model_root_dir}/{csv_name}_{log_time}_train.csv'
  df_train.to_csv(train_csv_path,mode='a')
  val_csv_path = f'{model_root_dir}/{csv_name}_{log_time}_val.csv'
  df_val.to_csv(val_csv_path,mode='a')