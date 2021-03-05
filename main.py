import logging
import time
import os
import tensorflow as tf
import numpy as np
import equation as eqn
import solver
from pprint import pformat
from utility import get_config, DictionaryUtility
#os.environ["CUDA_VISIBLE_DEVICES"]='2','3'
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('config_path', './configs/CovidMulti3.json',
                           """The path to load json file.""")
tf.app.flags.DEFINE_string('exp_dir', './data/debug',
                           """The directory of numerical experiment.""")


def main():
    os.makedirs(FLAGS.exp_dir, exist_ok=True)
    log_filename = os.path.join(FLAGS.exp_dir, 'log')

    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)-6s %(message)s', 
                        filemode='w', filename=log_filename)
    config = get_config(FLAGS.config_path)
    logging.info('Config file: %s' % FLAGS.config_path)
    logging.info('Experiment directory: %s' % FLAGS.exp_dir)
    logging.info(pformat(DictionaryUtility.to_dict(config)))

    bsde = getattr(eqn, config.eqn_config.eqn_name)(config.eqn_config)

    #mode='restore'  # for reading pre-trained model
    mode='train'  #for training model
    if mode=='train':
        with tf.Session() as sess:
            start_time = time.time()
            model = getattr(solver, config.nn_config.model_name)(config, bsde, sess)
            model.build()
            print('stage1')
            training_history = model.train_play(policy_func=model.old_policy)
            
            elapsed_time = time.time() - start_time
            logging.info('Elapsed time: %f s' % elapsed_time)
            np.savetxt(fname=os.path.join(FLAGS.exp_dir, 'train_hist.csv'), X=training_history,
                    delimiter=',', header=model.train_hist_header, comments='')  

            saver = tf.train.Saver(tf.global_variables())
            saver.save(sess,os.path.join(FLAGS.exp_dir, 'my_model'),global_step=1000)
            print(tf.global_variables(scope='forward')[0])
            print(tf.keras.backend.eval(tf.global_variables(scope='forward')[0]))

            dw_simulate, dw_sample, x_simulate,x_old_policy_simulate = bsde.simulate(
                    num_sample=config.nn_config.batch_size,
                    old_policy_func=model.old_policy)
            np.savez(file=os.path.join(FLAGS.exp_dir, 'siml_path.npz'),
                    dw_simulate=dw_simulate, dw_sample=dw_sample, x_simulate=x_simulate, x_old_policy_simulate=x_old_policy_simulate)



    
    if mode=='restore':

        with tf.Session() as sess:
            
            #sess.run(tf.global_variables_initializer())


            # print(tf.global_variables(scope='forward/player0/player0/batch_normalization/gamma:0')[0])
            # print(tf.keras.backend.eval(tf.global_variables(scope='forward/player0/player0/batch_normalization/gamma:0')[0]))
            
            new_saver = tf.train.import_meta_graph(os.path.join(FLAGS.exp_dir, 'my_model-1000.meta'))
            new_saver.restore(sess, tf.train.latest_checkpoint(os.path.join(FLAGS.exp_dir, './'))) 
            
            

            saved_dict = {}
            for x in tf.global_variables(scope='forward'):
                print(x.name)
                saved_dict[x.name] = x

            model = getattr(solver, config.nn_config.model_name)(config, bsde, sess)

            model.build_restore()

            for v in tf.global_variables(scope='aforward'):
                 if v.name[0]=='a':
                     sess.run(v.assign(saved_dict[v.name[1:]]))
            
            for v in tf.global_variables(scope='aforward'):
                 if v.name[0]=='a':
                     print(v.name,tf.keras.backend.eval(v))


            print(tf.global_variables(scope='forward/player0/policy_player0'))

            dw_simulate, dw_sample, x_simulate,x_old_policy_simulate = bsde.simulate(
                    num_sample=config.nn_config.batch_size,
                    old_policy_func=model.old_policy)
            np.savez(file=os.path.join(FLAGS.exp_dir, 'siml_path_restore.npz'),
                     dw_simulate=dw_simulate, dw_sample=dw_sample, x_simulate=x_simulate, x_old_policy_simulate=x_old_policy_simulate)


if __name__ == '__main__':
    main()
