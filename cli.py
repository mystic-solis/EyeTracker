import click

from commands.augmentor import augment
from commands.coco_to_yolo import dataset
from commands.prepare_file_struct import structure

from commands.remove_unlabeled import remove_unlabeled


@click.group()
def cli():
    """🔧 Утилита для работы с dicom и моделями машинного обучения"""
    pass

# Добавляем остальные команды в основную группу
cli.add_command(augment)
cli.add_command(dataset)
cli.add_command(structure)
cli.add_command(remove_unlabeled, name='remove')


if __name__ == '__main__':
    cli()