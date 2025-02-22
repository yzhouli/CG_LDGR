from model.prediction import Propagation_Transformer
from train.train import train


def main():
    source_path = f'data'
    mask_rate_li = [1, 0.5]
    learning_rate = 0.001
    train_rate = 0.4
    save_path = f'result_data/'
    epochs = 30

    model = Propagation_Transformer(
        embed_dim=3,
        depth=2,
        num_heads=1,
        num_classes=3,
        mask_rate_li=mask_rate_li,
        name="Information-Propagation-Prediction")

    train(path=source_path, model=model, learning_rate=learning_rate, save_path=save_path, epochs=epochs,
          train_rate=train_rate)


if __name__ == '__main__':
    main()
