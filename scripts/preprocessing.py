import glob, os, gc
import numpy as np
import pandas as pd
from IPython.core.display import HTML


def split_line(line: str)-> list:
  """
  line : str ('-59\t-86\t16.545591294123085\t')
  
  take line --> split --> if len 10 --> return
  else --> find index type --> find time --> (time+ type + data)
  --> repeat
  
  return ['1560830841636\tTYPE_ACCELEROMETER_UNCALIBRATED\t0.37109375\t-1.335968\t11.279083\n']
  """
  lines = []
  fields = [field for field in line.strip().split('\t')]
  idx_type = [i for i, d in enumerate(fields) if d in DATA_TYPES]
  # if correct data

  if len(fields) <= 10:
      return [line]    
  
  # bad   
  # len for next type    
  first_time_line = '\t'.join(fields[:idx_type[1]])
  end_first = len(first_time_line) - 13
  
  lines.append(line[:end_first])
  other = line[end_first:]
  # recurse
  for l in other.splitlines():
      lines += split_line(l)
  return lines


if __name__ == '__main__':
    
  PATH = '../input/train/'
  DATA_TYPES = ('TYPE_ACCELEROMETER',
                'TYPE_MAGNETIC_FIELD',
                'TYPE_GYROSCOPE',
                'TYPE_ROTATION_VECTOR',
                'TYPE_MAGNETIC_FIELD_UNCALIBRATED',
                'TYPE_GYROSCOPE_UNCALIBRATED',
                'TYPE_ACCELEROMETER_UNCALIBRATED',
                'TYPE_WIFI',
                'TYPE_BEACON',
                'TYPE_WAYPOINT')
  # need check works correct
  print('Test...')

  test = ['1560830841553', 'TYPE_BEACON', '89cb11b04122cef23388b0da06bd426c1f48a9b5',
          '4bd29af61db57eb8675b56f79c5ff4ae6f81c03a', 'ae38c9dac6a05831fe3016a1ed0519fb7f74feea',
          '-59', '-86', '16.545591294123085', '833d5531741cba213e7030266fb6d9e2ed4c2aca1560830841566',       
          'TYPE_BEACON', '89cb11b04122cef23388b0da06bd426c1f48a9b5', '4bd29af61db57eb8675b56f79c5ff4ae6f81c03a',
          'a77ff5d7252bab87e1eed09b2f29d47622ab9bd0', '-59', '-92', '27.752738449875483',
          '6c4ab94d11c2dfb10b9dd8fe35b59ab49cc2c5d61560830841579']

  test_line = '\t'.join(test)

  l0 = '1560830841553\tTYPE_BEACON\t' +\
        '89cb11b04122cef23388b0da06bd426c1f48a9b5\t' +\
        '4bd29af61db57eb8675b56f79c5ff4ae6f81c03a\t' +\
        'ae38c9dac6a05831fe3016a1ed0519fb7f74feea\t' +\
        '-59\t-86\t16.545591294123085\t' +\
        '833d5531741cba213e7030266fb6d9e2ed4c2aca'
  l1 = '1560830841566\tTYPE_BEACON\t' +\
        '89cb11b04122cef23388b0da06bd426c1f48a9b5\t' +\
        '4bd29af61db57eb8675b56f79c5ff4ae6f81c03a\t' +\
        'a77ff5d7252bab87e1eed09b2f29d47622ab9bd0\t' +\
        '-59\t-92\t27.752738449875483\t' +\
        '6c4ab94d11c2dfb10b9dd8fe35b59ab49cc2c5d61560830841579'

  assert split_line(test_line)[0] == l0, 'error not eq idx 0'
  assert split_line(test_line)[1] == l1, 'error not eq idx 1'

  wtf = test_line * 25

  outs = split_line(test_line)
  for idx in range(len(outs)):
      if idx % 2 == 0:
          assert split_line(test_line)[idx] == l0, f'error not eq idx {idx}'
      else:
          assert split_line(test_line)[idx] == l1, f'error not eq idx {idx}'

  test2 = ['TYPE_BEACON', '89cb11b04122cef23388b0da06bd426c1f48a9b5', '4bd29af61db57eb8675b56f79c5ff4ae6f81c03a',
  'e031005e7f6f7c2d5f91eb742b12f6a4b2e00434', '-59', '-87', '18.07763630020772',
  '3573eb6d74efbc27bbe52ff20b411e6bdbb5afe01560830842509', 
  'TYPE_ACCELEROMETER', '0.38067627', '-0.21069336', '9.275131']

  test_line2 = '\t'.join(test2)

  l2 = 'TYPE_BEACON\t89cb11b04122cef23388b0da06bd426c1f48a9b5\t' +\
        '4bd29af61db57eb8675b56f79c5ff4ae6f81c03a\te031005e7f6f7c2d5f91eb742b12f6a4b2e00434\t' +\
        '-59\t-87\t18.07763630020772\t3573eb6d74efbc27bbe52ff20b411e6bdbb5afe0'
  l3 = '1560830842509\tTYPE_ACCELEROMETER\t0.38067627\t-0.21069336\t9.275131'

  assert split_line(test_line2) == [l2, l3], 'error not eq'

  print('Correct')
    
  """
  
    kaggle current time script ~ 2.10h
    Size 9Gb
    
  """
  
  train_files = glob.glob('../Indoor_LN/input/train/*/*/*.txt')
  test_files = glob.glob('../Indoor_LN/input/test/*.txt')
  all_files = train_files + test_files
  print(train_files)
  max_len_col = 10
  for files in all_files:
      print(files)
      with open(files) as f:
          txt = f.readlines()
      # skip head
      len_head =[]
      for f in txt:
          if f.startswith('#'):
                len_head.append(f)
                if 'SiteID' in f:
                    # faind building id
                    site_id = f.split('\t')[1].split(':')[1]
      txt = txt[len(len_head):-1]
    
      data = []

      for line in txt:
          line_ = split_line(line)
          for l in line_:
            if len(fields) < max_len_col:
                fields = [field for field in l.strip().split('\t')]
#                 make all data eq len
                fields += [np.nan] * (max_len_col - len(fields))
            data.append(fields)
      columns = [f'columns_{i}' for i in range(1,len(fields)+1)]
      to_save = pd.DataFrame(data=data, columns=columns)   

      if files.split(os.path.sep)[-4] == 'train':        
          floor = files.split(os.path.sep)[-2]
          site = files.split(os.path.sep)[-3]
          arg = '../Indoor_LN/input/for_test/train'
          dirr = os.path.join(arg, site, floor)
      else:
          # save by building id
          dirr = os.path.join('../Indoor_LN/input/for_test/test', site_id)
      name = files.split(os.path.sep)[-1][:-4] 
      os.makedirs(dirr, exist_ok=True)    
      base = os.path.join(dirr, name + '.parquet')    
      to_save.to_parquet(base)

  gc.collect()