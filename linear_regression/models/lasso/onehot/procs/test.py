import sys

if len(sys.argv) != 2:
    print('')
    print(" Insufficient arguments passed!! \n Usage: %s Nx Ny dt" % sys.argv[0])
    print('')
    sys.exit(0)
# otherwise continue
me = int(sys.argv[1])
print(f'ME = {me}')