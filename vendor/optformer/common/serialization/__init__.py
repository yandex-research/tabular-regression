# Modified for vendoring: local namespace imports and optional dependency
# removal. Numerical serialization is retained; see VENDOR.md.

# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Entryway to common serializers."""

from vendor.optformer.common.serialization.base import Deserializer
from vendor.optformer.common.serialization.base import Serializer
from vendor.optformer.common.serialization.base import SerializerFactory
from vendor.optformer.common.serialization.primitive import PrimitiveSerializer
from vendor.optformer.common.serialization.primitive import PrimitiveType
from vendor.optformer.common.serialization.tokens import CartesianProductTokenSerializer
from vendor.optformer.common.serialization.tokens import IntegerTokenSerializer
from vendor.optformer.common.serialization.tokens import OneToManyTokenSerializer
from vendor.optformer.common.serialization.tokens import RepeatedUnitTokenSerializer
from vendor.optformer.common.serialization.tokens import StringTokenSerializer
from vendor.optformer.common.serialization.tokens import TokenSerializer
from vendor.optformer.common.serialization.tokens import UnitSequenceTokenSerializer
from vendor.optformer.common.serialization.tokens import UnitTokenSerializer
