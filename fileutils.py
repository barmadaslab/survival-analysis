import os 
import sys
import traceback

def log_tb_and_output_mssg(outdir, message):
    with open(outdir + '/error.log', 'w') as f:
        traceback.print_exc(file=f)
    sys.stderr.write(message)

def print_tb_and_mssg(exception, mssg=None, exit=True):
    traceback.print_tb(exception.__traceback__)
    if not mssg:
        print('\n' + str(exception.args[0]))
    else:
        print('\n' + mssg)
    if exit:
        sys.exit()

def mkdir(dir, exist_ok=True):
    try: 
        os.makedirs(dir, exist_ok=exist_ok)
    except OSError:
        raise
        # Can use/modify for logging error out.
        #exceptutils.log_tb_and_output_mssg('k:/rmdev', 'Unable to create the directory ' + dir + '\n')
