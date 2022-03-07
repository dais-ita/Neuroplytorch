.. _Datasets Page:

Datasets module
================================

Dataset classes 
----------------

BasicDataset
~~~~~~~~~~~~~~~~~

.. autoclass:: data.datasets.BasicDataset
   :members:
   :special-members:

EndEndDataset
~~~~~~~~~~~~~~~~~

.. autoclass:: data.datasets.EndEndDataset
   :members:
   :special-members:

Pytorch Lightning data modules
------------------------------

ReasoningDataModule
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: data.datasets.ReasoningDataModule
   :members:
   :special-members:

PerceptionWindowDataModule
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: data.datasets.PerceptionWindowDataModule
   :members:
   :special-members:

EndToEndDataModule
~~~~~~~~~~~~~~~~~~

.. autoclass:: data.datasets.EndToEndDataModule
   :members:
   :special-members:

EndToEndNoTestDataModule
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: data.datasets.EndToEndNoTestDataModule
   :members:
   :special-members:

MNISTDataModule
~~~~~~~~~~~~~~~~~

.. autoclass:: data.datasets.MNISTDataModule
   :members:
   :special-members:

EMNISTDataModule
~~~~~~~~~~~~~~~~~

.. autoclass:: data.datasets.EMNISTDataModule
   :members:
   :special-members:

Methods 
----------------------

.. autofunction:: data.datasets.get_datamodule
.. autofunction:: data.datasets.fetch_perception_data
.. autofunction:: data.datasets.fetch_perception_data_local