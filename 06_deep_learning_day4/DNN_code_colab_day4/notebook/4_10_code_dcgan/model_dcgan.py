import glob
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image

from dcgan_architecture import DCGAN_Discriminator, DCGAN_Generator


class DCGAN(object):
    def __init__(self, args_dict):
        self.build(args_dict)
        print("{:=^80}".format(" Model Initialization Done "))

    def set_optimizer(self, param_dict):
        lr = param_dict["lr"]
        beta_1 = param_dict["beta_1"]
        beta_2 = param_dict["beta_2"]
        epsilon = param_dict["epsilon"]
        self.g_optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon
        )
        self.d_optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon
        )

    def build(self, args_dict):
        self.exp_dir = Path.cwd() / "exp" / args_dict["exp_name"]
        Path(self.exp_dir).mkdir(exist_ok=True, parents=True)
        self.set_optimizer(args_dict["adam"])
        self.batch_size = args_dict["batch_size"]
        self.num_epoch = args_dict["num_epoch"]
        self.noize_dim = args_dict["noize_dim"]
        self.sample_freq = args_dict["sample_freq"]
        self.ckpt_freq = args_dict["ckpt_freq"]
        self.make_logs = args_dict["make_logs"]

        self.global_step = tf.Variable(
            0, trainable=False, dtype=tf.int64, name="global_step"
        )
        self.epoch = tf.Variable(0, trainable=False, dtype=tf.int64, name="epoch")
        if self.make_logs:
            log_dir = self.exp_dir / "logs"
            Path(log_dir).mkdir(exist_ok=True)
            tf.summary.experimental.set_step(self.global_step)
            self.summary_writer = tf.summary.create_file_writer(str(log_dir))

        generator = DCGAN_Generator(self.batch_size)
        discriminator = DCGAN_Discriminator(self.batch_size)
        self.G = generator.build()
        self.D = discriminator.build()

        ckpt_dir = self.exp_dir / "ckpt"
        Path(ckpt_dir).mkdir(exist_ok=True)
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.g_optimizer,
            discriminator_optimizer=self.d_optimizer,
            generator=self.G,
            discriminator=self.D,
            global_step=self.global_step,
            epoch=self.epoch,
        )
        self.manager = tf.train.CheckpointManager(
            self.checkpoint, directory=str(ckpt_dir), max_to_keep=50
        )
        # checkpointを呼び出して学習を再開する
        if args_dict["restore_ckpt"]:
            self.checkpoint.restore(self.manager.latest_checkpoint)
            if self.manager.latest_checkpoint:
                print("Restored ckpt from {}".format(self.manager.latest_checkpoint))
            else:
                print("Initializing from scratch.")
                raise ValueError("The pretrained checkpoint couldn't be found!")

    @tf.function
    def update_discriminator(self, noize, real_data):
        fake_data = self.G(noize)

        with tf.GradientTape() as d_tape:
            real_pred = self.D(real_data)
            fake_pred = self.D(fake_data)

            real_loss = tf.keras.losses.binary_crossentropy(
                tf.ones_like(real_pred), real_pred
            )
            fake_loss = tf.keras.losses.binary_crossentropy(
                tf.zeros_like(fake_pred), fake_pred
            )

            # batchの平均をとる
            real_loss = tf.math.reduce_mean(real_loss)
            fake_loss = tf.math.reduce_mean(fake_loss)
            adv_loss = real_loss + fake_loss

        d_grad = d_tape.gradient(adv_loss, sources=self.D.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_grad, self.D.trainable_variables))
        if self.make_logs:
            with self.summary_writer.as_default():
                tf.summary.scalar("d_loss", adv_loss)
            self.summary_writer.flush()

        return adv_loss

    @tf.function
    def update_generator(self, noize):
        with tf.GradientTape() as g_tape:
            fake_data = self.G(noize)
            fake_pred = self.D(fake_data)
            # max(log(D(x))) こちらの方が勾配消失に頑健
            fake_loss = tf.keras.losses.binary_crossentropy(
                tf.ones_like(fake_pred), fake_pred
            )
            # min(1-log(D(x)))
            # fake_loss = -tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_pred), fake_pred)

            # batchの平均をとる
            fake_loss = tf.math.reduce_mean(fake_loss)
        g_grad = g_tape.gradient(fake_loss, sources=self.G.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grad, self.G.trainable_variables))
        if self.make_logs:
            with self.summary_writer.as_default():
                tf.summary.scalar("g_loss", fake_loss)
        self.global_step.assign_add(1)
        return fake_loss

    def train(self, train_data):
        print("{:=^80}".format(" Training Start "))
        train_start_time = time.time()
        train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
        num_batch = len(train_data) // self.batch_size

        for e_idx in range(self.epoch.numpy(), self.num_epoch):
            noize = tf.random.normal((self.batch_size * num_batch, self.noize_dim))
            noize = (
                tf.data.Dataset.from_tensor_slices(noize)
                .shuffle(self.noize_dim)
                .batch(self.batch_size, drop_remainder=True)
            )
            shuffled_train_data = train_dataset.shuffle(len(train_data[0]))
            batch_train_data = shuffled_train_data.batch(
                self.batch_size, drop_remainder=True
            )
            zipped_train_dataset = tf.data.Dataset.zip((noize, batch_train_data))
            zipped_train_dataset = zipped_train_dataset.prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE
            )
            # 1step分のデータを取り出し
            for b_idx, (noize, train_data) in enumerate(zipped_train_dataset):
                batch_start_time = time.time()
                step = self.global_step.numpy()
                d_loss = self.update_discriminator(noize, train_data)
                g_loss = self.update_generator(noize)

                time_batch = time.time() - batch_start_time
                if b_idx < 1:
                    print(" epoch |   batch   |  time  |" "   D_loss   |   G_loss   ")
                print(
                    " {:5d} | {:4d}/{:4d} | {:6.2f} |"
                    " {:10.6f} | {:10.6f} ".format(
                        e_idx, b_idx + 1, num_batch, time_batch, d_loss, g_loss,
                    )
                )
            self.epoch.assign_add(1)
            if (e_idx + 1) % self.ckpt_freq == 0:
                self.generate_img(str(e_idx) + "epoch")
            if (e_idx + 1) % self.sample_freq == 0:
                print("{:=^80}".format(" Saving Checkpoint "))
                self.manager.save(checkpoint_number=step)
        total_time = time.time() - train_start_time
        print("{:=^80}".format(" Training Done "))
        print("Total Time: {:8.2f}".format(total_time))

    def generate_img(self, dir_name="infer", sample_times=1):
        print("Generate Images")
        save_dir = self.exp_dir / "samples" / dir_name
        Path(save_dir).mkdir(exist_ok=True, parents=True)
        for i in range(sample_times):
            noize = tf.random.normal((self.batch_size, self.noize_dim))
            generated_batch = self.G(noize)
            generated_batch = generated_batch.numpy()
            for j, generated_img in enumerate(generated_batch):
                rgb_image = (generated_img * 127.5 + 127.5).astype("uint8")
                pil_img = Image.fromarray(rgb_image)
                save_name = "generated_" + str(i) + "_" + str(j) + ".jpg"
                save_path = Path(save_dir) / save_name
                pil_img.save(save_path)
        print("Saved All Generated Data at " + str(save_dir))
