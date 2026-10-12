import sys
if __name__ == '__main__':
    try:
        from noise_aware.cli import main
        exit_code = main()
    except ModuleNotFoundError as error:
        packages = {'cv2': 'opencv-python', 'numpy': 'numpy', 'scipy': 'scipy', 'matplotlib': 'matplotlib'}
        package = packages.get(error.name)
        if package is None:
            raise
        print(f'Missing package: {package}. Install it in conda env before running.', file=sys.stderr)
        sys.exit(2)
    sys.exit(exit_code)
