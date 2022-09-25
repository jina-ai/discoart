import os
import threading
from threading import Thread

import torchvision.transforms.functional as TF
from docarray import DocumentArray, Document

from .helper import logger, get_output_dir


def _sample_thread(*args):
    t = Thread(
        target=_sample,
        args=(*args,),
    )
    t.start()
    return t


def _sample(
    sample,
    _nb,
    cur_t,
    da,
    da_gif,
    _handlers,
    j,
    loss_values,
    output_dir,
    is_sampling_done,
    is_save_step,
    is_save_gif,
    is_image_output,
    is_display_step,
    image_callback,
):
    with threading.Lock():
        is_sampling_done.clear()
        _display_html = []

        for k, image in enumerate(sample['pred_xstart']):  # batch_size
            image = TF.to_pil_image(image.add(1).div(2).clamp(0, 1))

            c = Document(
                tags={
                    '_status': {
                        'cur_t': cur_t,
                        'step': j,
                        'loss': loss_values[-1],
                        'minibatch_idx': k,
                    }
                }
            ).load_pil_image_to_datauri(image)

            if is_save_step:
                if is_image_output:
                    if cur_t == -1:
                        f_name = os.path.join(output_dir, f'{_nb}-done-{k}.png')
                    else:
                        f_name = os.path.join(output_dir, f'{_nb}-step-{j}-{k}.png')
                    c.save_uri_to_file(f_name)

                    if callable(image_callback):
                        image_callback(f_name)

                da[k].chunks.append(c)

            if is_save_gif and is_image_output:
                da_gif[k].chunks.append(c)

            # root doc always update with the latest progress
            da[k].uri = c.uri

            da[k].tags['_status'] = {
                'completed': cur_t == -1,
                'cur_t': cur_t,
                'step': j,
                'loss': loss_values,
            }

            _display_html.append(f'<img src="{c.uri}" alt="step {j} minibatch {k}">')

            if cur_t == -1:
                _handlers.completed.value += f'<br>seed: {da[k].tags["seed"]}<br><img src="{c.uri}" alt="step {j} minibatch {k}"><br>'

        if is_display_step:
            _handlers.preview.value = '<br>\n'.join(_display_html)

        logger.debug('sample and plot is done')
        is_sampling_done.set()


def _save_progress_thread(*args):
    t = Thread(
        target=_save_progress,
        args=(*args,),
    )
    t.start()
    return t


def _save_progress(da, da_gif, _nb, output_dir, fps, size_ratio):
    with threading.Lock():
        try:
            for idx, d in enumerate(da):
                if d.chunks:
                    # only print the first image of the minibatch in progress
                    d.chunks.plot_image_sprites(
                        os.path.join(output_dir, f'{_nb}-progress-{idx}.png'),
                        skip_empty=True,
                        show_index=True,
                        keep_aspect_ratio=True,
                    )
            for idx, d_gif in enumerate(da_gif):
                if d_gif.chunks and fps > 0:
                    d_gif.chunks.save_gif(
                        os.path.join(output_dir, f'{_nb}-progress-{idx}.gif'),
                        skip_empty=True,
                        show_index=True,
                        duration=1000 // fps,
                        size_ratio=size_ratio,
                    )
            logger.debug('progress are stored in as png and gif')
        except ValueError:
            logger.debug('can not plot progress into sprite image and gif')


def _persist_thread(
    da_batches, name_docarray, is_busy_evs, is_sampling_done, is_completed
):
    for fn, idle_ev in zip((_local_save, _cloud_push), is_busy_evs):
        t = Thread(
            target=fn,
            args=(da_batches, name_docarray, idle_ev, is_sampling_done, is_completed),
        )
        t.start()
        yield t


def _local_save(
    da_batches: DocumentArray,
    name: str,
    is_busy_event: threading.Event,
    is_sampling_done: threading.Event,
    force: bool = False,
) -> None:
    if 'DISCOART_OPTOUT_LOCAL_BACKUP' in os.environ:
        return
    if is_busy_event.is_set() and not force:
        logger.debug(f'another save is running, skipping')
        return
    is_sampling_done.wait()
    is_busy_event.set()
    try:
        pb_path = os.path.join(get_output_dir(name), f'da.protobuf.lz4')
        da_batches.save_binary(pb_path)
        logger.debug(f'local backup to {pb_path}')
    except Exception as ex:
        logger.debug(f'local backup failed: {ex}')
    is_busy_event.clear()


def _cloud_push(
    da_batches: DocumentArray,
    name: str,
    is_busy_event: threading.Event,
    is_sampling_done: threading.Event,
    force: bool = False,
) -> None:
    if 'DISCOART_OPTOUT_CLOUD_BACKUP' in os.environ:
        return
    if is_busy_event.is_set() and not force:
        logger.debug(f'another cloud backup is running, skipping')
        return
    is_sampling_done.wait()
    is_busy_event.set()

    try:
        da_batches.push(name)
        logger.debug(f'cloud backup to {name}')
    except Exception as ex:
        logger.debug(f'cloud backup failed: {ex}')
    is_busy_event.clear()
