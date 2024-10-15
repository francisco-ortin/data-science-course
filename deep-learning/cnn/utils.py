import PIL.Image, PIL.ImageFont, PIL.ImageDraw
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds


def _read_image_tfds(image: tf.Tensor, label: tf.Tensor) -> (tf.Tensor, (tf.Tensor, tf.Tensor)):
    """
    Places a 28x28 image randomly in 75x75 image, writing a bounding box around it
    :param image: the image
    :param label: the label
    :return: the image and the label with the bounding box
    """
    xmin = tf.random.uniform((), 0, 48, dtype=tf.int32)
    ymin = tf.random.uniform((), 0, 48, dtype=tf.int32)
    image = tf.reshape(image, (28, 28, 1,))
    image = tf.image.pad_to_bounding_box(image, ymin, xmin, 75, 75)
    image = tf.cast(image, tf.float32) / 255.0
    xmin = tf.cast(xmin, tf.float32)
    ymin = tf.cast(ymin, tf.float32)
    xmax = (xmin + 28) / 75
    ymax = (ymin + 28) / 75
    xmin = xmin / 75
    ymin = ymin / 75
    return image, (tf.one_hot(label, 10), [xmin, ymin, xmax, ymax])


def get_train_val_dataset(batch_size: int, val_split: float = 0.15) -> (tf.data.Dataset, tf.data.Dataset):
    """
    Downloads the training MNIST dataset from tensorflow datasets
    and splits it into training and validation datasets
    :param batch_size: the batch size
    :param val_split: the validation split
    :return: the training and validation datasets
    """
    dataset, ds_info = tfds.load('mnist', split='train', as_supervised=True, try_gcs=True, with_info=True)
    num_examples = ds_info.splits['train'].num_examples
    dataset = dataset.map(_read_image_tfds, num_parallel_calls=16)
    dataset = dataset.shuffle(5000, reshuffle_each_iteration=True)
    dataset = dataset.repeat()
    train_dataset = dataset.take(train_size := int((1-val_split) * num_examples))
    val_dataset = dataset.skip(train_size).take(int(val_split * num_examples))
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    train_dataset = train_dataset.prefetch(-1)
    val_dataset = val_dataset.batch(batch_size, drop_remainder=True)
    val_dataset = val_dataset.prefetch(-1)
    return train_dataset, val_dataset


def get_test_dataset() -> tf.data.Dataset:
    """
    Downloads the test MNIST dataset from tensorflow datasets
    :return: the test dataset
    """
    dataset = tfds.load("mnist", split="test", as_supervised=True, try_gcs=True)
    dataset = dataset.map(_read_image_tfds, num_parallel_calls=16)
    dataset = dataset.batch(10000, drop_remainder=True)  # 10000 items in eval dataset, all in one batch
    dataset = dataset.repeat()  # Mandatory for Keras for now
    return dataset


def dataset_to_numpy_util(train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, test_ds: tf.data.Dataset, N: int) -> \
        (np.array, np.array, np.array, np.array, np.array, np.array, np.array, np.array, np.array):
    """
    Converts the three raw tf datasets into NumPy's arrays
    :param train_ds: the training dataset
    :param val_ds: the validation dataset
    :param test_ds: the test dataset
    :param N: the number of elements to convert
    :return: the numpy arrays (train_digits, train_labels, train_bboxes, val_digits, val_labels, val_bboxes,
    test_digits, test_labels, test_bboxes)
    """
    batch_train_ds = train_ds.unbatch().batch(N)
    if tf.executing_eagerly():
        for train_digits, (train_labels, train_bboxes) in batch_train_ds:
            train_digits = train_digits.numpy()
            train_labels = train_labels.numpy()
            train_bboxes = train_bboxes.numpy()
            break
        for val_digits, (val_labels, val_bboxes) in val_ds:
            val_digits = val_digits.numpy()
            val_labels = val_labels.numpy()
            val_bboxes = val_bboxes.numpy()
            break
        for test_digits, (test_labels, test_bboxes) in test_ds:
            test_digits = test_digits.numpy()
            test_labels = test_labels.numpy()
            test_bboxes = test_bboxes.numpy()
            break
    train_labels = np.argmax(train_labels, axis=1)
    val_labels = np.argmax(val_labels, axis=1)
    test_labels = np.argmax(test_labels, axis=1)
    return (train_digits, train_labels, train_bboxes,
            val_digits, val_labels, val_bboxes,
            test_digits, test_labels, test_bboxes)


def _draw_bounding_boxes_on_image_array(image: np.array, boxes: np.array, color=[], thickness=1, display_str_list=())\
        -> np.array:
    """
    Draws the bounding box in an image array
    :param image: the image
    :param boxes: the boxes
    :param color: the color
    :param thickness: the thickness
    :param display_str_list: the display string list
    :return: the image array with the bounding boxes
    """
    image_pil = PIL.Image.fromarray(image)
    rgbimg = PIL.Image.new('RGBA', image_pil.size)
    rgbimg.paste(image_pil)
    _draw_bounding_boxes_on_image(rgbimg, boxes, color, thickness, display_str_list)
    return np.array(rgbimg)


def _draw_bounding_boxes_on_image(image: PIL.Image, boxes: np.array, color=[], thickness=1, display_str_list=()) \
        -> None:
    """
    Draws bounding boxes in an image
    :param image: the image
    :param boxes: the boxes
    :param color: the color
    :param thickness: the thickness
    :param display_str_list: the display string list
    """
    boxes_shape = boxes.shape
    if not boxes_shape:
        return
    if len(boxes_shape) != 2 or boxes_shape[1] != 4:
        raise ValueError('Input must be of size [N, 4]')
    for i in range(boxes_shape[0]):
        _draw_bounding_box_on_image(image, boxes[i, 1], boxes[i, 0], boxes[i, 3], boxes[i, 2], color[i], thickness,
                                    display_str_list[i])


def _draw_bounding_box_on_image(image: PIL.Image, ymin: float, xmin: float, ymax: float, xmax: float,
                                color: str = 'red', thickness: int = 1, display_str_list: list[str] = None,
                                use_normalized_coordinates:bool = True) -> None:
    """
    Draws a bounding box in an image
    :param image: the image
    :param ymin: the minimum y
    :param xmin: the minimum x
    :param ymax: the maximum y
    :param xmax: the maximum x
    :param color: the color
    :param thickness: the thickness
    :param display_str_list: the display string list
    :param use_normalized_coordinates: whether to use normalized
    """
    draw = PIL.ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, ymin, xmax, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=thickness, fill=color)


def display_digits_with_boxes(digits: np.array, predictions: np.array, labels: np.array, pred_bboxes: np.array,
                              bboxes: np.array, title: str) -> None:
    """
    Shows predictions and bounding boxes
    :param digits: the digits
    :param predictions: the predictions
    :param labels: the labels
    :param pred_bboxes: the predicted bounding boxes
    :param bboxes: the bounding boxes
    :param title: the title of the figure
    """
    n = 10
    indexes = np.random.choice(len(predictions), size=n)
    n_digits = digits[indexes]
    n_predictions = predictions[indexes]
    n_labels = labels[indexes]
    if len(pred_bboxes) > 0:
        n_pred_bboxes = pred_bboxes[indexes,:]
    if len(bboxes) > 0:
        n_bboxes = bboxes[indexes,:]
    n_digits = n_digits * 255.0
    n_digits = n_digits.reshape(n, 75, 75)
    fig = plt.figure(figsize=(20, 4))
    plt.title(title)
    plt.yticks([])
    plt.xticks([])
    for i in range(10):
        ax = fig.add_subplot(1, 10, i+1)
        bboxes_to_plot = []
        if len(pred_bboxes) > i:
            bboxes_to_plot.append(n_pred_bboxes[i])
        if len(bboxes) > i:
            bboxes_to_plot.append(n_bboxes[i])
        img_to_draw = _draw_bounding_boxes_on_image_array(image=n_digits[i], boxes=np.asarray(bboxes_to_plot), color=['red', 'green'], display_str_list=["true", "pred"])
        plt.xlabel(n_predictions[i])
        plt.xticks([])
        plt.yticks([])
        if n_predictions[i] != n_labels[i]:
            ax.xaxis.label.set_color('red')
        plt.imshow(img_to_draw)




