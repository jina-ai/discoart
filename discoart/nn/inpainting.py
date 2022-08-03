def wrap_inpaint(diff, inpaint_tensor, mask_tensor, noise):
    def inpaint_sampling(f):
        def func(*args, **kwargs):
            result = f(*args, **kwargs)
            curr_t = args[-1]
            curr_image = diff.q_sample(inpaint_tensor, curr_t, noise)
            result["sample"] = curr_image * mask_tensor + result["sample"] * (1 - mask_tensor)
            return result

        return func

    diff.ddim_sample = inpaint_sampling(diff.ddim_sample)
    diff.plms_sample = inpaint_sampling(diff.plms_sample)