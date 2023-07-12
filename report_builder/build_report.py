from bs4 import BeautifulSoup
import os
import sys
import time
import argparse
from shutil import copyfile, copytree, rmtree

from .utils import *
from ..models.load_model import load_model

base_path = '/home/pierre/Documents/PHD/phd_utils/report_builder'

parser = argparse.ArgumentParser(description='Detection training')
parser.add_argument('--path', type=str,
                    help='Path to the model to be evaluated.', required=True)
parser.add_argument('--store-in-zoo', action='store_false',
                    help='Move model in zoo')
parser.add_argument('--new-name', type=str, default=None,
                    help='New name for network.')
parser.add_argument('--copy-training', action='store_false',
                    help='Copy training logs next to network. ')
parser.add_argument('--no-details', action='store_true',
                    help='Write report without model description.')
parser.add_argument('--no-eval', action='store_true',
                    help='Write report without model evaluation.')


args = parser.parse_args()
zoo_path = '/home/pierre/Documents/PHD/model_zoo'

print('Starting building report.')
tic = time.time()

model_path = args.path
model_name = args.new_name if args.new_name is not None else model_path.split('/')[-1]

save_path = os.path.join(zoo_path, model_name.split('.')[0])
if not os.path.isdir(save_path):
    print('No model with name {} found, creating a new folder in zoo.'.format(model_name))
    os.mkdir(save_path)
    os.mkdir(os.path.join(save_path, 'logs'))
    os.mkdir(os.path.join(save_path, 'fig'))
else:
    print('Model already in zoo: overwritting? (y/n)')
    ans = input()
    if ans != 'y':
        sys.exit()

if args.store_in_zoo:
    copyfile(model_path, os.path.join(save_path, model_name))

## Load model
model = load_model(model_path)
config = model.config

if hasattr(config, 'log_path') and config.log_path is not None and args.copy_training:
    if os.path.isdir(os.path.join(save_path, 'logs')):
        rmtree(os.path.join(save_path, 'logs'))
    copytree(config.log_path, os.path.join(save_path, 'logs'))
if hasattr(config, 'log_path') and config.log_path is not None:
    log_path = config.log_path
    date = '_'.join(log_path.split('/')[-1].split('_')[1:])[:-7]
else:
    log_path = 'n/a'
    date = 'n/a'

if not args.no_details:
    print('Short description about experiment:\n')
    details_str = input()
else:
    details_str = ''

build_embeddings(model, save_path)
# build_proposals(model, save_path)

if not args.no_eval:
    classes_train, classes_test, metrics = compute_metrics(model, shots=[1,3,5,10], num_val_episodes=15)
else:
    metrics = {'shots': [1, 3, 5, 10], 
                'train-train': {1: 0.0, 3: 0.0, 5: 0.0, 10: 0.0}, 
                'train-test': {1: 0.0, 3: 0.0, 5: 0.0, 10: 0.0}, 
                'test-train': {1: 0.0, 3: 0.0, 5: 0.0, 10: 0.0}, 
                'test-test': {1: 0.0, 3: 0.0, 5: 0.0, 10: 0.0}}
    classes_train, classes_test = None, None

template_path = os.path.join(base_path, 'template.html')
soup = BeautifulSoup(open(template_path), 'html.parser')

### Update title
title = soup.find('title')
title.append('Report {}'.format(model_name))

name_tag = soup.find('h1', {'id': 'name'})
date_tag = soup.find('h3', {'id': 'date'})
path_tag = soup.find('h3', {'id': 'path'})
details = soup.find('p', {'id': 'details'})

name_tag.append(model_name)
date_tag.append(date)
path_tag.append(log_path)
details.append(details_str)

### Update Config
config_container = soup.find('div', {'id':'config-container'})

table_template = '''<table id='config-table'">
                                <tr>
                                    <th colspan="2">{}</th>
                                </tr> 
                                <tr>
                                    <th>Parameter</th>
                                    <th>Value</th>
                                </tr>

                    </table>'''

grouped_config = config.to_grouped_dict()
config_dict = config.to_dict()
for section, params in grouped_config.items():
    soup_table = BeautifulSoup(table_template.format(section), 'html.parser')
    config_table = soup_table.find('table')
    for param in params:
        new_row = soup_table.new_tag('tr')
        col1 = soup_table.new_tag('td')
        col1.append(str(param))
        col2 = soup_table.new_tag('td')
        col2.append(str(config_dict[param]))
        new_row.append(col1)
        new_row.append(col2)
        config_table.append(new_row)
    config_container.append(soup_table)

### Update embeddings section
rpn_emb = soup.find('img', {'id':'rpn-emb'})
rpn_hist = soup.find('img', {'id':'rpn-hist'})
rpn_proto = soup.find('img', {'id':'rpn-proto'})
roi_emb = soup.find('img', {'id':'roi-emb'})
roi_hist = soup.find('img', {'id':'roi-hist'})
roi_proto = soup.find('img', {'id':'roi-proto'})

rpn_emb['src'] = os.path.join(save_path, 'fig/rpn_emb.png')
rpn_hist['src'] = os.path.join(save_path, 'fig/rpn_hist.png')
rpn_proto['src'] = os.path.join(save_path, 'fig/rpn_proto.png')

roi_emb['src'] = os.path.join(save_path, 'fig/roi_emb.png')
roi_hist['src'] = os.path.join(save_path, 'fig/roi_hist.png')
roi_proto['src'] = os.path.join(save_path, 'fig/roi_proto.png')

### Update proposal section
prop_full = soup.find('img', {'id':'proposals-full'})
prop_train = soup.find('img', {'id':'proposals-train'})
prop_test = soup.find('img', {'id':'proposals-test'})

prop_full['src'] = os.path.join(save_path, 'fig/proposals_full.png')
prop_train['src'] = os.path.join(save_path, 'fig/proposals_train.png')
prop_test['src'] = os.path.join(save_path, 'fig/proposals_test.png')

### Update eval table
eval_table = soup.find('table', {'id': 'eval-table'})
for header, values in metrics.items():
    cell_tag = 'th' if header == 'shots' else 'td'
    new_row = soup.new_tag('tr')
    header_cell = soup.new_tag(cell_tag)
    header_cell.append(header)
    new_row.append(header_cell)
    for v in (values.values() if header != 'shots' else values):
        new_cell = soup.new_tag(cell_tag)
        new_cell.append('{:.3f}'.format(float(v)) if header != 'shots' else '{}'.format(v))
        new_row.append(new_cell)
    eval_table.append(new_row)

train_class_text = soup.find('p', {'id': 'train-classes'})
train_class_text.append(str(classes_train))
test_class_text = soup.find('p', {'id': 'test-classes'})
test_class_text.append(str(classes_test))

### Write report in HTML
with open(os.path.join(save_path, '{}.html'.format(model_name.split('.')[0])), 'wb') as f_output:
    f_output.write(soup.prettify('utf-8')) 

copyfile(os.path.join(base_path, 'styles.css'), os.path.join(save_path, 'styles.css'))

print('Total time: {:.2f}s.'.format(time.time() - tic)) 