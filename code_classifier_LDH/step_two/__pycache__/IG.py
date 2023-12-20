import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sys



def interpolate_images(baseline,
                       image,
                       alphas):
  alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
  baseline_x = tf.expand_dims(baseline, axis=0)
  input_x = tf.expand_dims(image, axis=0)
  delta = input_x - baseline_x
  images = baseline_x +  alphas_x * delta
  return images


def compute_gradients(model, images, target_class_idx):
  with tf.GradientTape() as tape:
    tape.watch(images)
    logits = model(images)
    probs = logits[:, target_class_idx]
  return tape.gradient(probs, images)


def integral_approximation(gradients):
  grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
  integrated_gradients = tf.math.reduce_mean(grads, axis=0)
  return integrated_gradients


@tf.function
def integrated_gradients(model,
                         baseline,
                         image,
                         target_class_idx,
                         m_steps=50,
                         batch_size=32):
  alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)

  gradient_batches = tf.TensorArray(tf.float32, size=m_steps+1)
    
  for alpha in tf.range(0, len(alphas), batch_size):
    from_ = alpha
    to = tf.minimum(from_ + batch_size, len(alphas))
    alpha_batch = alphas[from_:to]

   
    interpolated_path_input_batch = interpolate_images(baseline=baseline,
                                                       image=image,
                                                       alphas=alpha_batch)

   
    gradient_batch = compute_gradients(model=model, images=interpolated_path_input_batch,
                                       target_class_idx=target_class_idx)
    
   
    gradient_batches = gradient_batches.scatter(tf.range(from_, to), gradient_batch)    
  
  total_gradients = gradient_batches.stack()

  avg_gradients = integral_approximation(gradients=total_gradients)

  integrated_gradients = (image - baseline) * avg_gradients

  return integrated_gradients

def convergence_check(model, attributions, baseline, input, target_class_idx):

  baseline_prediction = model(tf.expand_dims(baseline, 0))

  baseline_score = baseline_prediction[0][target_class_idx]


  input_prediction = model(tf.expand_dims(input, 0))

  input_score = input_prediction[0][target_class_idx]

  ig_score = tf.math.reduce_sum(attributions)
  delta = ig_score - (input_score - baseline_score)

  try:
    # Test your IG score is <= 5% of the input minus baseline score.
    tf.debugging.assert_near(ig_score, (input_score - baseline_score), rtol=0.05)
    tf.print('Approximation accuracy within 5%.', output_stream=sys.stdout)
  except tf.errors.InvalidArgumentError:
    tf.print('Increase or decrease m_steps to increase approximation accuracy.', output_stream=sys.stdout)
  
  tf.print('Baseline score: {:.3f}'.format(baseline_score))
  tf.print('Input score: {:.3f}'.format(input_score))
  tf.print('IG score: {:.3f}'.format(ig_score))     
  tf.print('Convergence delta: {:.3f}'.format(delta))

def plot_img_attributions(model,
                          baseline,
                          image,
                          target_class_idx,
                          m_steps=50,
                          cmap=None,
                          overlay_alpha=0.4,
                          top_prob=0.0,
                          top_label="",
                          meta={}):


  attributions = integrated_gradients(model=model,
                                      baseline=baseline,
                                      image=image,
                                      target_class_idx=target_class_idx,
                                      m_steps=m_steps)

  convergence_check(model=model,
                    attributions=attributions,
                    baseline=baseline,
                    input=image,
                    target_class_idx=target_class_idx)


  attribution_mask = tf.reduce_sum(tf.math.abs(attributions), axis=-1)

  fig, axs = plt.subplots(nrows=1, ncols=3, squeeze=False, figsize=(9,4))

  file_name = meta["file_name"]
  v = meta["v"]
  position = ""
  mode = meta["mode"]


  position = f'P{meta["position_index"]}'


  axs[0, 0].set_title('Original image')
  axs[0, 0].imshow(image)
  axs[0, 0].axis('off')

  axs[0, 1].set_title('Attribution mask')
  axs[0, 1].imshow(attribution_mask, cmap=cmap)
  axs[0, 1].axis('off')

  axs[0, 2].set_title('Overlay')
  axs[0, 2].imshow(attribution_mask, cmap=cmap)
  axs[0, 2].imshow(image, alpha=overlay_alpha)
  axs[0, 2].axis('off')

  save_name = f'{file_name}-{mode}-{position}-{top_label}-{top_prob:0.1%}'
  fig.suptitle(save_name, fontweight='bold')
  plt.tight_layout()
  # plt.show()  # this is needed to block the process
  plt.savefig(f'{meta["save_dir"]}/{save_name}.jpeg')
  # close figure by plt.close(fig), it won't be displayed
  plt.close(fig)
  return fig


def main_ig(model, img_tensor, target_class_idx, prediction, meta):

    # print("\n\n======== main_ig called ============")
    # print("target_class_idx: ", target_class_idx)
    top_prob = np.max(prediction[0])
    grading = np.array(['1 ', '2 ', '3 ', '4 '])
    top_label = grading[target_class_idx]
    # print("img_tensor: ", img_tensor.shape, img_tensor.dtype, img_tensor[0][0])
    # ============ Constants ===================
    baseline = tf.zeros(shape=(150,150,3))

    # if needs to Visualizing gradient saturation
    visualize_grad_saturation = False
    if visualize_grad_saturation:
        m_steps = 50
        alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1) # Generate m_steps intervals for integral_approximation() below.

        interpolated_images = interpolate_images(
            baseline=baseline,
            image=img_tensor,
            alphas=alphas)

        # ### Compute Gradients

        path_gradients = compute_gradients(
            model=model,
            images=interpolated_images,
            target_class_idx=target_class_idx)
        # print("path_gradients: ", path_gradients.shape)
        # print(np.max(path_gradients), np.min(path_gradients))
        
        # Visualize the gradient saturation
        pred = model(interpolated_images)
        pred_proba = pred[:, target_class_idx]

        plt.figure(figsize=(10, 4))
        ax1 = plt.subplot(1, 2, 1)
        ax1.plot(alphas, pred_proba)
        ax1.set_title('Target class predicted probability over alpha')
        ax1.set_ylabel('model p(target class)')
        ax1.set_xlabel('alpha')
        ax1.set_ylim([0, 1])

        ax2 = plt.subplot(1, 2, 2)
        # Average across interpolation steps
        average_grads = tf.reduce_mean(path_gradients, axis=[1, 2, 3])
        # Normalize gradients to 0 to 1 scale. E.g. (x - min(x))/(max(x)-min(x))
        average_grads_norm = (average_grads-tf.math.reduce_min(average_grads))/(tf.math.reduce_max(average_grads)-tf.reduce_min(average_grads))
        ax2.plot(alphas, average_grads_norm)
        ax2.set_title('Average pixel gradients (normalized) over alpha')
        ax2.set_ylabel('Average pixel gradients')
        ax2.set_xlabel('alpha')
        ax2.set_ylim([0, 1]);
        plt.show()

    # =========== main program ================
    # ## Visualize Attributions
    _ = plot_img_attributions(model=model,
                            image=img_tensor,
                            baseline=baseline,
                            target_class_idx=target_class_idx,
                            m_steps=240,
                            cmap=plt.cm.inferno,
                            overlay_alpha=0.4,
                            top_prob=top_prob,
                            top_label=top_label,
                            meta=meta)
