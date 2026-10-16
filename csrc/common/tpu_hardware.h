/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef TORCH_TPU_CSRC_COMMON_TPU_HARDWARE_H_
#define TORCH_TPU_CSRC_COMMON_TPU_HARDWARE_H_

#include <string_view>

#include "absl/status/statusor.h"

namespace torch_tpu {

// Counts the TPU devices on this host that are exposed via VFIO, by scanning
// the PCI devices in `pci_devices_dir` for known TPU vendor/device IDs and
// checking that each device's IOMMU group exists in `vfio_dir`. Does not
// initialize the TPU runtime.
//
// Returns an error only if `pci_devices_dir` cannot be opened or read. Devices
// whose sysfs entries cannot be read are skipped. The count does NOT include:
// - TPUs whose PCI device ID is not in the known list (e.g. future
//   generations);
// - TPUs whose IOMMU group is not exposed under `vfio_dir`, which is normal in
//   containers that expose only the VFIO groups allocated to them;
// - TPUs bound to a non-VFIO driver (e.g. reached via /dev/accel*).
// So a count of 0 does not mean that the host has no TPUs, and this function is
// unsuitable for deciding that TPUs are absent; see HasTpuDeviceNodes().
//
// Mirrors `_scan_pci_tpus()` in torch_tpu/_internal/utils/hardware.py; keep the
// two in sync.
absl::StatusOr<int> CountTpuDevicesViaSysfs(
    std::string_view pci_devices_dir = "/sys/bus/pci/devices",
    std::string_view vfio_dir = "/dev/vfio");

// Returns whether any device node through which libtpu could reach a TPU
// exists:
// - a VFIO group node `<dev_dir>/vfio/<number>` (not the container node
//   `<dev_dir>/vfio/vfio`);
// - a VFIO per-device (iommufd) node, i.e. any entry of
//   `<dev_dir>/vfio/devices/`, e.g. `vfio0`;
// - an accel node `<dev_dir>/accel*`, `<dev_borg_dir>/accel*`,
//   `<dev_dir>/accel/*` or `<dev_borg_dir>/accel/*` (the layout of the kernel's
//   compute accelerator subsystem, e.g. /dev/accel/accel0) that is a character
//   device whose major is registered in `proc_devices` (i.e. backed by a
//   driver).
// Unlike CountTpuDevicesViaSysfs(), this does not depend on PCI device IDs, so
// it also covers unknown TPU generations. Returns true if a directory exists
// but cannot be listed, or if an accel character device exists but
// `proc_devices` cannot be read, since the absence of TPU device nodes cannot
// be established then. Does not initialize the TPU runtime.
//
// A process without any of these nodes cannot use a TPU through the default
// device paths. Hosts with other VFIO or accel devices (e.g. a NIC bound to
// VFIO) also yield true, so true does not mean that a TPU is present.
[[nodiscard]] bool HasTpuDeviceNodes(
    std::string_view dev_dir = "/dev",
    std::string_view dev_borg_dir = "/dev_borg",
    std::string_view proc_devices = "/proc/devices");

}  // namespace torch_tpu

#endif  // TORCH_TPU_CSRC_COMMON_TPU_HARDWARE_H_
