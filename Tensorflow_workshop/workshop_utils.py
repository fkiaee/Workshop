import tensorflow as tf
import scipy.misc as misc
import numpy as np
from fnmatch import fnmatch
import os 
import pickle
import matplotlib.pyplot as plt
# reset the default grapg to empty and stop having to restart the kernel repeatedly

def ImageProducer_from_CSV(filename_queue):
    line_reader = tf.TextLineReader()
    key, line = line_reader.read(filename_queue)
     # line_batch or line (depending if you want to batch)
    filename, label = tf.decode_csv(line,record_defaults=[tf.constant([],dtype=tf.string),tf.constant([],dtype=tf.int32)],field_delim=' ')
    file_contents = tf.read_file(filename)
    image = tf.image.decode_png(file_contents) 
    image.set_shape([32, 32, 3])
    image = tf.cast(image, tf.float32)   
    image = tf.image.per_image_standardization(image)
    return image, label


def CreateCSV(images_path,experience):     
    # merge files and labels in one csv file
    pattern = '*.PNG'
    subdirs = [subdirs for path, subdirs, files in os.walk(images_path) if len(subdirs)>0][0]
    class_label = 0
    for sub in subdirs:
        lists = [os.path.join(path,name) for path, subdirs, files in os.walk(os.path.join(images_path,sub)) for name in files if fnmatch(name, pattern)]
        #print(lists)
        labels = class_label*np.ones((len(lists),),int)
        with open('./'+experience + '_'+ 'train.csv', 'a') as export:
            for ii in range(int(0.4*len(lists))):
                export.write(lists[ii]+' ')
                export.write(str(labels[ii])+'\n')    
        with open('./'+experience + '_'+ 'validate.csv', 'a') as export:
            for ii in range(int(0.4*len(lists)),int(len(lists))):
                export.write(lists[ii]+' ')
                export.write(str(labels[ii])+'\n') 
        class_label = class_label + 1    


def ImageProducer_from_Bin(filename_queue): 
    label_bytes = 1; 
    height = 32; width = 32; depth = 3
    image_bytes = height * width * depth
    record_bytes = label_bytes + image_bytes
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    key, value = reader.read(filename_queue)
    record_bytes = tf.decode_raw(value, tf.uint8)
    label_byte_slices = tf.slice(record_bytes, [0], [label_bytes]);
    label = tf.cast(label_byte_slices, tf.int32)
    image = tf.slice(record_bytes, [label_bytes], [image_bytes])#tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),[depth,height,width])
    image = tf.cast(image, tf.float32)   
    image = tf.reshape(image,[1,image_bytes])  
    depth_major = tf.reshape(image,[depth,height,width])
    image = tf.transpose(depth_major, [1, 2, 0])  
    image = tf.image.per_image_standardization(image)
    return image, label    
    
def CreateBin(images_path,experience):     
    files_list = [os.path.join(path,name) for path, subdirs, files in os.walk(images_path) for name in files if fnmatch(name,'*.png')]
    R_data = [[np.reshape(misc.imread(filenames,0)[:,:,0],-1)] for filenames in files_list]
    G_data = [[np.reshape(misc.imread(filenames,0)[:,:,1],-1)] for filenames in files_list]
    B_data = [[np.reshape(misc.imread(filenames,0)[:,:,2],-1)] for filenames in files_list]
    #print([name for path, subdirs, files in os.walk(images_path) for name in files if fnmatch(name,'*.png')])
    label = np.zeros((np.shape(R_data)[0],1),int)
    subdirs = [subdirs for path, subdirs, files in os.walk(images_path) if len(subdirs)>0][0]     # get the subfolders (classes)
    class_label = 0; st_index = 0;
    for sub in subdirs:
        lists = [os.path.join(path,name) for path, subdirs, files in os.walk(os.path.join(images_path,sub)) for name in files if fnmatch(name, '*.png')]
        #print(lists)
        label[st_index:st_index+len(lists)] = class_label*np.ones((len(lists),1),int)
        st_index = st_index + len(lists)
        class_label = class_label + 1
    #print(label)
    label = label.astype('uint8')
    
    R_data = np.array(np.squeeze(R_data))
    G_data = np.array(np.squeeze(G_data))
    B_data = np.array(np.squeeze(B_data))
    outdata = np.concatenate((label,R_data,G_data,B_data), axis = 1)
    
    num_samples = np.shape(outdata)[0];
    
    #train_samples = random.sample(range(num_samples),num_samples/2)
    #test_samples = list(set(range(num_samples))-set(train_samples));
    Indexs = np.arange(num_samples)
    np.random.shuffle(Indexs)
    
    train_samples = Indexs[:int(0.6*num_samples)]
    test_samples = Indexs[int(0.6*num_samples):]
    
    
    temp = outdata[train_samples,:]
    temp.tofile(experience +'_train.bin')
    print('train data saved')
    
    temp = outdata[test_samples,:]
    temp.tofile(experience +'_validate.bin')
    print('test data saved')

    
def network(inpt, keep_prob, classes_num):
    print("setting up the network ...")
    coef = 0.2
    h = 3; w = 3; in_c = 3; out_c = 32; gain = coef*np.sqrt(2/(h*w*in_c))
    c1_f = tf.Variable(gain*tf.random_normal([h, w, in_c, out_c]), name='c1_f')  
    c1_b = tf.Variable(tf.constant(0.0, shape=[out_c]), name='c1_b')
    conv1 = tf.nn.conv2d(inpt, c1_f, strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.bias_add(conv1, c1_b)
    tf.add_to_collection('weight_decay', tf.nn.l2_loss(c1_f))
    tf.add_to_collection('Transferable_vars',c1_f)
    tf.add_to_collection('Transferable_vars',c1_b)
    relue1 = tf.nn.relu(conv1)
    Pool1 = tf.nn.max_pool(relue1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    h = 3; w = 3; in_c = 32; out_c = 64; gain = coef*np.sqrt(2/(h*w*in_c))
    c2_f = tf.Variable(gain*tf.random_normal([h, w, in_c, out_c]), name='c2_f')  
    c2_b = tf.Variable(tf.constant(0.0, shape=[out_c]), name='c2_b')
    conv2 = tf.nn.conv2d(Pool1, c2_f, strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.bias_add(conv2, c2_b)
    tf.add_to_collection('weight_decay', tf.nn.l2_loss(c2_f))
    tf.add_to_collection('Transferable_vars',c2_f)
    tf.add_to_collection('Transferable_vars',c2_b)
    relue2 = tf.nn.relu(conv2)
    Pool2 = tf.nn.max_pool(relue2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='VALID')
    drop2 = tf.nn.dropout(Pool2, keep_prob=keep_prob)    

    h = 3; w = 3; in_c = 64; out_c = 128; gain = coef*np.sqrt(2/(h*w*in_c))
    c3_f = tf.Variable(gain*tf.random_normal([h, w, in_c, out_c]), name='c3_f')  
    c3_b = tf.Variable(tf.constant(0.0, shape=[out_c]), name='c3_b')
    conv3 = tf.nn.conv2d(drop2, c3_f, strides=[1, 1, 1, 1], padding='SAME')
    conv3 = tf.nn.bias_add(conv3, c3_b)
    tf.add_to_collection('weight_decay', tf.nn.l2_loss(c3_f))
    tf.add_to_collection('Transferable_vars',c3_f)
    tf.add_to_collection('Transferable_vars',c3_b)
    relue3 = tf.nn.relu(conv3)
    Pool3 = tf.nn.max_pool(relue3, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='VALID')
    drop3 = tf.nn.dropout(Pool3, keep_prob=keep_prob)    


    h = 1; w = 1; in_c = 128; out_c = classes_num; gain = coef*np.sqrt(2/(h*w*in_c))
    c_f_classifier = tf.Variable(gain*tf.random_normal([h, w, in_c, out_c]),  name='c_f_classifier')  
    c_b_classifier = tf.Variable(tf.constant(0.0, shape=[out_c]), name='c_b_classifier')
    classifier = tf.nn.conv2d(drop3, c_f_classifier, strides=[1, 1, 1, 1], padding='SAME')
    classifier = tf.nn.bias_add(classifier, c_b_classifier)
    classifier = tf.reshape(classifier, [ classifier.get_shape().as_list()[0], -1])
    tf.add_to_collection('weight_decay', tf.nn.l2_loss(c_f_classifier))
    tf.add_to_collection('Classifier_vars',c_f_classifier)
    tf.add_to_collection('Classifier_vars',c_b_classifier)
#    pred = tf.reshape(pred, [-1, 128 * 4 * 4])
#    pred = tf.nn.relu(tf.matmul(pred, w_f1) + b_f1)
#    pred = tf.matmul(pred, w_out)

    weight_decay_sum = tf.add_n(tf.get_collection('weight_decay'))
    TRANSFERABLE_VARIABLES = tf.get_collection('Transferable_vars')
    CLASSIFIER_VARIABLES = tf.get_collection('Classifier_vars')

    return classifier, TRANSFERABLE_VARIABLES, CLASSIFIER_VARIABLES, weight_decay_sum


def network_train_validate(experience, checkpoint_dir, phase, classes_num, netpre_path = None):
    tf.reset_default_graph()
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    keep_prob = tf.placeholder("float")
    
    num_samples = 500*classes_num; ckp_write_period = 20;
    batch_size = 100
    shuffle_train = True; im_visualize_flag = True
    # Create a queue that produces the filenames to read.
        
    #filenames = [experience + '_'+ phase + '.bin']
    #filename_queue = tf.train.string_input_producer(filenames, num_epochs=80, shuffle=shuffle_train, seed = 2)
    #example, label = ImageProducer_from_Bin(filename_queue);
    
    filenames = [experience + '_'+ phase + '.csv']
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=80, shuffle=shuffle_train, seed = 2)
    example, label = ImageProducer_from_CSV(filename_queue);

    min_fraction_of_examples_in_queue = 0.4; min_queue_examples = int(num_samples * min_fraction_of_examples_in_queue)
    
    if phase == 'train':
        example_batch, label_batch = tf.train.shuffle_batch([example, label], batch_size=batch_size, num_threads=16, capacity=min_queue_examples + 3 * batch_size, min_after_dequeue=min_queue_examples)
        feed_dict ={keep_prob:0.7}
    else:
        example_batch,label_batch = tf.train.batch([example, label], batch_size)
        feed_dict ={keep_prob:1.}
    label_batch = tf.reshape(label_batch, [batch_size])
    label_batch = tf.cast(label_batch, tf.int64)    

    classifier, TRANSFERABLE_VARIABLES, CLASSIFIER_VARIABLES, weight_decay_sum = network(example_batch, keep_prob, classes_num)
    variable_names = [vv.name for vv in TRANSFERABLE_VARIABLES]
    # variable_names = ['c1_f','c1_b','c2_f','c2_b','c3_f','c3_b'] 
    
    # Define the loss function
    probs = tf.nn.softmax(classifier)
    #loss = tf.reduce_mean(-tf.reduce_sum(label * tf.log(probs), [3]))
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits = classifier,
                            labels = label_batch)) + tf.multiply(0.0001,weight_decay_sum)
    # Use a gradient descent as optimization method
    learning_rate = 0.001;
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    #train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(loss)

    if netpre_path:
        learning_rate1 = 1e-5; learning_rate2 = 0.001;
        opt1 = tf.train.AdamOptimizer(learning_rate1)
        opt2 = tf.train.AdamOptimizer(learning_rate2)
        grads = tf.gradients(loss,TRANSFERABLE_VARIABLES+CLASSIFIER_VARIABLES)
        grads1 = grads[:len(TRANSFERABLE_VARIABLES)]
        grads2 = grads[len(TRANSFERABLE_VARIABLES):len(TRANSFERABLE_VARIABLES)+len(CLASSIFIER_VARIABLES)]
        train_op1 = opt1.apply_gradients(zip(grads1,TRANSFERABLE_VARIABLES))
        train_op2 = opt2.apply_gradients(zip(grads2, CLASSIFIER_VARIABLES))
        train_op = tf.group(train_op1, train_op2)   
    
    # We define the prediction as the index of the highest output
    prediction = tf.argmax(classifier, 1)
    
    # This node checks if the prediction is equal to the actual answer
    ispredcorrect = tf.equal(prediction, label_batch)
    accuracy = tf.reduce_mean(tf.cast(ispredcorrect, 'float'))
    
    # Only used for vizualisation purposes
    loss_disp = tf.summary.scalar("Cross_entropy", loss)
    acc_disp = tf.summary.scalar("Accuracy", accuracy)
    merged_display =tf.summary.merge_all()
    
    
    sess = tf.Session();
    # Write graph infos to the specified file
    summary_writer = tf.summary.FileWriter(checkpoint_dir+"/tflogs_"+phase, sess.graph)
    
    #tf.add_to_collection('non_trainable_variables')#tf.all_variables()
    saver = tf.train.Saver(tf.get_collection('non_trainable_variables')+tf.trainable_variables(),max_to_keep=None);
    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
    sess.run(init); 
    if netpre_path:
        weights_file = open(os.path.join(netpre_path, 'transferable_weights.pkl'), 'rb')
        transferable_weights = pickle.load(weights_file)
        weights_file.close()
        for var, weight in zip(TRANSFERABLE_VARIABLES, transferable_weights):
            var.load(weight, sess)
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir) 
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)      
    ##=========begin training======================================
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess,coord)
    correct = 0;  count = 0.; max_steps = 1000
    try:
        while not coord.should_stop():
            if im_visualize_flag:
                f, a = plt.subplots(5, 1, figsize=(10, 10), squeeze=False)
                im = sess.run(example_batch, feed_dict=feed_dict)
                for i in range(5):
                    a[i][0].imshow(im[i,:,:,:].astype('uint8'))
                plt.show()
                im_visualize_flag = False
            if phase == 'train':
                sess.run(train_op, feed_dict=feed_dict)
                if count % ckp_write_period == 0:
                    checkpoint_path = os.path.join(checkpoint_dir, "model_ckpt%s.ckpt"%(count+1))
                    saver.save(sess, checkpoint_path)
                    transferable_weights = sess.run(TRANSFERABLE_VARIABLES)                        
                    print('These variables are saving in a .pkl file',variable_names)
                    filename = os.path.join(checkpoint_dir,'transferable_weights.pkl')
                    file1 = open(filename, 'wb')
                    pickle.dump(transferable_weights,file1)
                    file1.close() 

            summary_result = sess.run(merged_display, feed_dict=feed_dict)
            summary_writer.add_summary(summary_result, count)
            summary_writer.flush()
            count += 1
            correct += np.sum(sess.run(accuracy,feed_dict=feed_dict))
            Acc_Perf = float(correct)/ (count)
            print("Accuracy at each step:",Acc_Perf)            
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()
        coord.join(threads)                     

