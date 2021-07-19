from attrdict import AttrDict

from ..vocab import NULL


class SlotBase(AttrDict):
    REQUIRED_ARGS = [('name', str), ('is_null', bool), ('is_nullable', bool)]

    def __init__(self, **kwargs):
        for ra in SlotBase.REQUIRED_ARGS:
            key = ra[0]
            value = kwargs[key]

        super(SlotBase, self).__init__(**kwargs)

    def __str__(self):
        return f"Slot({self.name})"

    def __repr__(self):
        return self.__str__()

    def set_property(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v


class TemplateBase:
    def __init__(self, slots, **kwargs):
        self.name = kwargs.pop('name', self.__class__.__name__)
        self.slots = slots

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __lt__(self, other):
        return self.name < other.name

    @property
    def slot_num(self):
        return len(self.slots)


class TemplateCollectionBase:
    def __init__(self, templates, vocab, masker, slot_names):
        self.templates = templates
        self.vocab = vocab
        self.masker = masker
        self.__slot_names = slot_names

    def get(self, idx_or_str):
        if isinstance(idx_or_str, str):
            tmp = [t for t in self.templates if t.name == idx_or_str]
            if len(tmp) == 0:
                return None
            else:
                return tmp[0]
        elif isinstance(idx_or_str, int):
            return self.templates[idx_or_str]

    def generate(self, action):
        action = action[:, 1:]

        action_strs_spl = self.vocab.to_str_batch(action, int_cast=True)

        action_strs = []

        for a in action_strs_spl:
            a = [w.replace(NULL, '') for (idx, w) in enumerate(a)]
            a = ' '.join(a).strip()
            action_strs.append(a)

        return action_strs

    def generate_mask(self, template_idx, masker_params=None):
        template = self.templates[template_idx]

        masker_params = masker_params or {}

        return self.masker.generate_mask(template, **masker_params)

    def generate_mask_batch(self, template_indices, masker_params=None):
        templates = [self.templates[i] for i in template_indices]

        masker_params = masker_params or {}

        return self.masker.generate_mask_batch(templates, **masker_params)

    def generate_mask_all(self, masker_params=None):
        return [self.generate_mask(i, masker_params) for i in range(self.template_num)]

    def generate_mask_all_batch(self, masker_params=None, batch_size=None):
        if masker_params is not None:
            tmp_key = list(masker_params.keys())[0]
            batch_size = len(masker_params[tmp_key])
        else:
            assert batch_size is not None

        all_template_indices = list(range(self.template_num))

        return [self.generate_mask_batch(all_template_indices, masker_params) for i in range(batch_size)]

    def convert_mask_to_str(self, mask):
        return self.masker.check_mask(mask)

    def is_generable(self, target, verbose=True, **kwargs):
        if isinstance(target, str):
            target = target.split()

        for w in target:
            if not self.vocab.has(w):
                return False, {}

        if len(target) < 4:
            target += ['Null'] * (4 - len(target))

        status = {}
        
        for t in sorted(self.templates):
            status[t.name] = {s: {'n': 0, 'tokens': []} for s in ['v', 'o1', 'p', 'o2']}
            masks, masks_per_masker = self.masker.generate_mask(t, return_all=True, **kwargs)

            success = True
            for w, (slot_name, mask) in zip(target, masks.items()):

                if slot_name == NULL:
                    status['t.name'][slot_name]['n'] = 1
                    status['t.name'][slot_name]['tokens'].append('Null')
                    continue

                i = self.vocab.to_id(w)

                available_tokens = self.vocab.to_str_list(mask.nonzero(as_tuple=False).view(-1).tolist())
                status[t.name][slot_name] = \
                    {'n': len(available_tokens), 'tokens': available_tokens}

                if not mask[i]:
                    success = False

            status[t.name]['success'] = success

        generable_templates = [(t, d) for (t, d) in status.items() if d['success']]

        success = len(generable_templates) != 0

        return success, status

    @property
    def template_num(self):
        return len(self.templates)

    @property
    def vocab_size(self):
        return self.vocab.size

    @property
    def slot_num(self):
        return max([t.slot_num for t in self.templates])

    @property
    def slot_names(self):
        return self.__slot_names


