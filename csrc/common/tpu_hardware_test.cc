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
#include "csrc/common/tpu_hardware.h"

#include <ftw.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <ios>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "csrc/common/error_utils.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace torch_tpu {
namespace {

using absl_testing::IsOkAndHolds;
using absl_testing::StatusIs;
using testing::HasSubstr;

// Scoped temporary directory mimicking /sys/bus/pci/devices, /dev (including
// /dev/vfio, /dev/vfio/devices and /dev/accel) and /dev_borg.
class ScopedFakeHardwareDir {
 public:
  ScopedFakeHardwareDir() {
    std::string tmpl =
        absl::StrCat(testing::TempDir(), "/fake_tpu_hardware_XXXXXX");
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    const char* dir = mkdtemp(buf.data());
    ABSL_CHECK_NE(dir, nullptr);
    root_ = dir;
    pci_devices_dir_ = absl::StrCat(root_, "/pci_devices");
    dev_dir_ = absl::StrCat(root_, "/dev");
    vfio_dir_ = absl::StrCat(dev_dir_, "/vfio");
    vfio_devices_dir_ = absl::StrCat(vfio_dir_, "/devices");
    dev_borg_dir_ = absl::StrCat(root_, "/dev_borg");
    proc_devices_ = absl::StrCat(root_, "/proc_devices");
  }

  ~ScopedFakeHardwareDir() { RemoveTree(root_); }

  ScopedFakeHardwareDir(const ScopedFakeHardwareDir&) = delete;
  ScopedFakeHardwareDir& operator=(const ScopedFakeHardwareDir&) = delete;

  [[nodiscard]] const std::string& pci_devices_dir() const {
    return pci_devices_dir_;
  }
  [[nodiscard]] const std::string& dev_dir() const { return dev_dir_; }
  [[nodiscard]] const std::string& vfio_dir() const { return vfio_dir_; }
  [[nodiscard]] const std::string& dev_borg_dir() const {
    return dev_borg_dir_;
  }
  [[nodiscard]] const std::string& proc_devices() const {
    return proc_devices_;
  }

  // Creates an empty PCI devices directory.
  void AddPciDevicesDir() { MakeDir(pci_devices_dir_); }

  // Adds a file named `name` to the fake /dev directory.
  void AddDevNode(std::string_view name) {
    MakeDir(dev_dir_);
    WriteFile(absl::StrCat(dev_dir_, "/", name), "");
  }

  // Adds a file named `name` to the fake /dev_borg directory.
  void AddDevBorgNode(std::string_view name) {
    MakeDir(dev_borg_dir_);
    WriteFile(absl::StrCat(dev_borg_dir_, "/", name), "");
  }

  // Adds an entry named `name` to the fake /dev directory that resolves to the
  // real device node `target`. Tests cannot create device nodes themselves.
  void AddDevNodeLinkTo(std::string_view name, std::string_view target) {
    MakeDir(dev_dir_);
    AddLink(absl::StrCat(dev_dir_, "/", name), target);
  }

  // Like AddDevNodeLinkTo(), for the fake /dev_borg directory.
  void AddDevBorgNodeLinkTo(std::string_view name, std::string_view target) {
    MakeDir(dev_borg_dir_);
    AddLink(absl::StrCat(dev_borg_dir_, "/", name), target);
  }

  // Writes the fake /proc/devices, listing `char_majors` as the registered
  // character device majors.
  void WriteProcDevices(absl::Span<const int> char_majors) {
    std::string contents = "Character devices:\n";
    for (const int major : char_majors) {
      absl::StrAppend(&contents, absl::StrFormat("%3d", major), " drv\n");
    }
    // A block device major must not count as a character device major.
    absl::StrAppend(&contents, "\nBlock devices:\n  7 loop\n");
    WriteFile(proc_devices_, contents);
  }

  // Adds a PCI device entry. If `iommu_group` is set, adds an `iommu_group`
  // symlink to it, as the kernel does.
  void AddPciDevice(std::string_view address, std::string_view vendor,
                    std::string_view device,
                    std::optional<std::string_view> iommu_group) {
    MakeDir(pci_devices_dir_);
    const std::string device_dir = absl::StrCat(pci_devices_dir_, "/", address);
    MakeDir(device_dir);
    WriteFile(absl::StrCat(device_dir, "/vendor"), absl::StrCat(vendor, "\n"));
    WriteFile(absl::StrCat(device_dir, "/device"), absl::StrCat(device, "\n"));
    if (iommu_group.has_value()) {
      const std::string target =
          absl::StrCat("../../../kernel/iommu_groups/", *iommu_group);
      ABSL_CHECK_EQ(symlink(target.c_str(),
                            absl::StrCat(device_dir, "/iommu_group").c_str()),
                    0);
    }
  }

  // Adds a VFIO group node for `iommu_group`.
  void AddVfioGroup(std::string_view iommu_group) {
    MakeDir(dev_dir_);
    MakeDir(vfio_dir_);
    WriteFile(absl::StrCat(vfio_dir_, "/", iommu_group), "");
  }

  // Creates the fake /dev/vfio/devices directory, which holds the VFIO
  // per-device (iommufd) nodes, and returns its path.
  const std::string& AddVfioDevicesDir() {
    MakeDir(dev_dir_);
    MakeDir(vfio_dir_);
    MakeDir(vfio_devices_dir_);
    return vfio_devices_dir_;
  }

  // Adds a VFIO per-device (iommufd) node named `name`, e.g. "vfio0".
  void AddVfioDeviceNode(std::string_view name) {
    WriteFile(absl::StrCat(AddVfioDevicesDir(), "/", name), "");
  }

  // Adds an entry named `name` to the fake /dev/accel directory that resolves
  // to the real device node `target`.
  void AddDevAccelDirNodeLinkTo(std::string_view name,
                                std::string_view target) {
    MakeDir(dev_dir_);
    const std::string accel_dir = absl::StrCat(dev_dir_, "/accel");
    MakeDir(accel_dir);
    AddLink(absl::StrCat(accel_dir, "/", name), target);
  }

  // Like AddDevAccelDirNodeLinkTo(), for the fake /dev_borg/accel directory.
  void AddDevBorgAccelDirNodeLinkTo(std::string_view name,
                                    std::string_view target) {
    MakeDir(dev_borg_dir_);
    const std::string accel_dir = absl::StrCat(dev_borg_dir_, "/accel");
    MakeDir(accel_dir);
    AddLink(absl::StrCat(accel_dir, "/", name), target);
  }

 private:
  static void MakeDir(const std::string& path) {
    ABSL_CHECK(mkdir(path.c_str(), 0700) == 0 || errno == EEXIST);
  }

  static void AddLink(const std::string& path, std::string_view target) {
    ABSL_CHECK_EQ(symlink(std::string(target).c_str(), path.c_str()), 0);
  }

  static void WriteFile(const std::string& path, std::string_view contents) {
    std::ofstream file(path, std::ios::binary);
    file.write(contents.data(), contents.size());
    ABSL_CHECK(file.good());
  }

  static int RemoveEntry(const char* path, const struct stat* /*sb*/,
                         int /*typeflag*/, struct FTW* /*ftwbuf*/) {
    return remove(path);
  }

  static void RemoveTree(const std::string& root) {
    // FTW_DEPTH visits children first; FTW_PHYS avoids following symlinks.
    nftw(root.c_str(), &RemoveEntry, /*nopenfd=*/16, FTW_DEPTH | FTW_PHYS);
  }

  std::string root_;
  std::string pci_devices_dir_;
  std::string dev_dir_;
  std::string vfio_dir_;
  std::string vfio_devices_dir_;
  std::string dev_borg_dir_;
  std::string proc_devices_;
};

// Returns the major number of the character device `path`.
[[nodiscard]] int CharDeviceMajor(const char* path) {
  struct stat st;
  ABSL_CHECK_EQ(stat(path, &st), 0);
  ABSL_CHECK(S_ISCHR(st.st_mode));
  return static_cast<int>(major(st.st_rdev));
}

TEST(CountTpuDevicesViaSysfsTest, ReturnsErrorWithoutPciDevicesDir) {
  ScopedFakeHardwareDir fake;
  EXPECT_THAT(CountTpuDevicesViaSysfs(fake.pci_devices_dir(), fake.vfio_dir()),
              StatusIs(error::kUnavailable,
                       HasSubstr("cannot open the PCI devices directory")));
}

TEST(CountTpuDevicesViaSysfsTest, ReturnsZeroWithEmptyPciDevicesDir) {
  ScopedFakeHardwareDir fake;
  fake.AddPciDevicesDir();
  EXPECT_THAT(CountTpuDevicesViaSysfs(fake.pci_devices_dir(), fake.vfio_dir()),
              IsOkAndHolds(0));
}

TEST(CountTpuDevicesViaSysfsTest, IgnoresNonGoogleVendor) {
  ScopedFakeHardwareDir fake;
  fake.AddPciDevice("0000:00:04.0", "0x8086", "0x005e", "7");
  fake.AddVfioGroup("7");
  EXPECT_THAT(CountTpuDevicesViaSysfs(fake.pci_devices_dir(), fake.vfio_dir()),
              IsOkAndHolds(0));
}

TEST(CountTpuDevicesViaSysfsTest, CountsTpuDevicesWithVfioGroup) {
  ScopedFakeHardwareDir fake;
  fake.AddPciDevice("0000:00:04.0", "0x1ae0", "0x0063", "0");
  fake.AddPciDevice("0000:00:05.0", "0x1ae0", "0x006f", "1");
  fake.AddVfioGroup("0");
  fake.AddVfioGroup("1");
  EXPECT_THAT(CountTpuDevicesViaSysfs(fake.pci_devices_dir(), fake.vfio_dir()),
              IsOkAndHolds(2));
}

TEST(CountTpuDevicesViaSysfsTest, IgnoresTpuDeviceWithoutVfioGroup) {
  ScopedFakeHardwareDir fake;
  // Group 3 has no /dev/vfio node.
  fake.AddPciDevice("0000:00:04.0", "0x1ae0", "0x0076", "3");
  // No iommu_group symlink at all.
  fake.AddPciDevice("0000:00:05.0", "0x1ae0", "0x0076", std::nullopt);
  fake.AddVfioGroup("4");
  EXPECT_THAT(CountTpuDevicesViaSysfs(fake.pci_devices_dir(), fake.vfio_dir()),
              IsOkAndHolds(0));
}

TEST(CountTpuDevicesViaSysfsTest, IgnoresUnknownGoogleDeviceId) {
  ScopedFakeHardwareDir fake;
  fake.AddPciDevice("0000:00:04.0", "0x1ae0", "0x0001", "0");
  fake.AddVfioGroup("0");
  EXPECT_THAT(CountTpuDevicesViaSysfs(fake.pci_devices_dir(), fake.vfio_dir()),
              IsOkAndHolds(0));
}

TEST(CountTpuDevicesViaSysfsTest, CountsAllKnownTpuDeviceIds) {
  ScopedFakeHardwareDir fake;
  const std::vector<std::string> device_ids = {
      "0x0062",  // TPU v5p
      "0x0063",  // TPU v5e
      "0x006f",  // TPU v6e
      "0x0076",  // TPU v7
      "0x007c",  // TPU v8t
      "0x0083",  // TPU v8i
  };
  for (size_t i = 0; i < device_ids.size(); ++i) {
    const std::string group = absl::StrCat(i);
    fake.AddPciDevice(absl::StrCat("0000:00:0", i, ".0"), "0x1ae0",
                      device_ids[i], group);
    fake.AddVfioGroup(group);
  }
  EXPECT_THAT(CountTpuDevicesViaSysfs(fake.pci_devices_dir(), fake.vfio_dir()),
              IsOkAndHolds(static_cast<int>(device_ids.size())));
}

TEST(HasTpuDeviceNodesTest, ReturnsFalseWithoutDevDirs) {
  ScopedFakeHardwareDir fake;
  EXPECT_FALSE(HasTpuDeviceNodes(fake.dev_dir(), fake.dev_borg_dir(),
                                 fake.proc_devices()));
}

TEST(HasTpuDeviceNodesTest, IgnoresUnrelatedNodes) {
  ScopedFakeHardwareDir fake;
  fake.WriteProcDevices({CharDeviceMajor("/dev/null")});
  fake.AddDevNodeLinkTo("null", "/dev/null");
  // The VFIO container node exists whenever the VFIO module is loaded, with or
  // without any device bound to VFIO.
  fake.AddVfioGroup("vfio");
  fake.AddDevBorgNodeLinkTo("null", "/dev/null");
  EXPECT_FALSE(HasTpuDeviceNodes(fake.dev_dir(), fake.dev_borg_dir(),
                                 fake.proc_devices()));
}

TEST(HasTpuDeviceNodesTest, DetectsVfioGroupNode) {
  ScopedFakeHardwareDir fake;
  fake.AddVfioGroup("vfio");
  fake.AddVfioGroup("12");
  EXPECT_TRUE(HasTpuDeviceNodes(fake.dev_dir(), fake.dev_borg_dir(),
                                fake.proc_devices()));
}

TEST(HasTpuDeviceNodesTest, DetectsAccelNodeWithRegisteredMajor) {
  ScopedFakeHardwareDir fake;
  fake.WriteProcDevices({CharDeviceMajor("/dev/null")});
  fake.AddDevNodeLinkTo("accel0", "/dev/null");
  EXPECT_TRUE(HasTpuDeviceNodes(fake.dev_dir(), fake.dev_borg_dir(),
                                fake.proc_devices()));
}

TEST(HasTpuDeviceNodesTest, DetectsDevBorgAccelNodeWithRegisteredMajor) {
  ScopedFakeHardwareDir fake;
  fake.WriteProcDevices({CharDeviceMajor("/dev/null")});
  fake.AddDevBorgNodeLinkTo("accel3", "/dev/null");
  EXPECT_TRUE(HasTpuDeviceNodes(fake.dev_dir(), fake.dev_borg_dir(),
                                fake.proc_devices()));
}

TEST(HasTpuDeviceNodesTest, IgnoresAccelNodeWithUnregisteredMajor) {
  ScopedFakeHardwareDir fake;
  // Some static /dev trees hold many accel nodes without a driver behind them.
  fake.WriteProcDevices({CharDeviceMajor("/dev/null") + 1});
  fake.AddDevNodeLinkTo("accel0", "/dev/null");
  fake.AddDevBorgNodeLinkTo("accel1", "/dev/null");
  EXPECT_FALSE(HasTpuDeviceNodes(fake.dev_dir(), fake.dev_borg_dir(),
                                 fake.proc_devices()));
}

TEST(HasTpuDeviceNodesTest, IgnoresAccelEntryThatIsNotCharDevice) {
  ScopedFakeHardwareDir fake;
  fake.WriteProcDevices({CharDeviceMajor("/dev/null")});
  fake.AddDevNode("accel0");
  fake.AddDevBorgNode("accel1");
  EXPECT_FALSE(HasTpuDeviceNodes(fake.dev_dir(), fake.dev_borg_dir(),
                                 fake.proc_devices()));
}

TEST(HasTpuDeviceNodesTest, IgnoresAccelBlockDevice) {
  const char* block_device = nullptr;
  for (const char* const path : {"/dev/loop0", "/dev/sda", "/dev/vda",
                                 "/dev/nvme0n1", "/dev/ram0", "/dev/xda"}) {
    struct stat st;
    if (stat(path, &st) == 0 && S_ISBLK(st.st_mode)) {
      block_device = path;
      break;
    }
  }
  if (block_device == nullptr) {
    GTEST_SKIP() << "no block device found to link to";
  }
  struct stat st;
  ASSERT_EQ(stat(block_device, &st), 0);
  ScopedFakeHardwareDir fake;
  fake.WriteProcDevices({static_cast<int>(major(st.st_rdev))});
  fake.AddDevNodeLinkTo("accel0", block_device);
  EXPECT_FALSE(HasTpuDeviceNodes(fake.dev_dir(), fake.dev_borg_dir(),
                                 fake.proc_devices()));
}

TEST(HasTpuDeviceNodesTest, ReturnsTrueForAccelNodeIfProcDevicesUnreadable) {
  ScopedFakeHardwareDir fake;
  // No fake /proc/devices file.
  fake.AddDevNodeLinkTo("accel0", "/dev/null");
  EXPECT_TRUE(HasTpuDeviceNodes(fake.dev_dir(), fake.dev_borg_dir(),
                                fake.proc_devices()));
}

TEST(HasTpuDeviceNodesTest, DetectsVfioPerDeviceNode) {
  ScopedFakeHardwareDir fake;
  // The VFIO cdev (iommufd) interface exposes /dev/vfio/devices/vfioN instead
  // of, or in addition to, the group nodes.
  fake.AddVfioGroup("vfio");
  fake.AddVfioDeviceNode("vfio0");
  EXPECT_TRUE(HasTpuDeviceNodes(fake.dev_dir(), fake.dev_borg_dir(),
                                fake.proc_devices()));
}

TEST(HasTpuDeviceNodesTest, IgnoresEmptyVfioDevicesDir) {
  ScopedFakeHardwareDir fake;
  fake.AddVfioDevicesDir();
  EXPECT_FALSE(HasTpuDeviceNodes(fake.dev_dir(), fake.dev_borg_dir(),
                                 fake.proc_devices()));
}

TEST(HasTpuDeviceNodesTest, ReturnsTrueWhenVfioDevicesDirCannotBeListed) {
  if (geteuid() == 0) {
    GTEST_SKIP() << "root can list directories without read permission";
  }
  ScopedFakeHardwareDir fake;
  const std::string& vfio_devices_dir = fake.AddVfioDevicesDir();
  ASSERT_EQ(chmod(vfio_devices_dir.c_str(), 0), 0);
  EXPECT_TRUE(HasTpuDeviceNodes(fake.dev_dir(), fake.dev_borg_dir(),
                                fake.proc_devices()));
  ASSERT_EQ(chmod(vfio_devices_dir.c_str(), 0700), 0);
}

TEST(HasTpuDeviceNodesTest, DetectsAccelDirNodeWithRegisteredMajor) {
  ScopedFakeHardwareDir fake;
  // The kernel's compute accelerator subsystem creates /dev/accel/accelN.
  fake.WriteProcDevices({CharDeviceMajor("/dev/null")});
  fake.AddDevAccelDirNodeLinkTo("accel0", "/dev/null");
  EXPECT_TRUE(HasTpuDeviceNodes(fake.dev_dir(), fake.dev_borg_dir(),
                                fake.proc_devices()));
}

TEST(HasTpuDeviceNodesTest, DetectsDevBorgAccelDirNodeWithRegisteredMajor) {
  ScopedFakeHardwareDir fake;
  fake.WriteProcDevices({CharDeviceMajor("/dev/null")});
  fake.AddDevBorgAccelDirNodeLinkTo("accel2", "/dev/null");
  EXPECT_TRUE(HasTpuDeviceNodes(fake.dev_dir(), fake.dev_borg_dir(),
                                fake.proc_devices()));
}

TEST(HasTpuDeviceNodesTest, IgnoresAccelDirNodeWithUnregisteredMajor) {
  ScopedFakeHardwareDir fake;
  fake.WriteProcDevices({CharDeviceMajor("/dev/null") + 1});
  fake.AddDevAccelDirNodeLinkTo("accel0", "/dev/null");
  fake.AddDevBorgAccelDirNodeLinkTo("accel1", "/dev/null");
  EXPECT_FALSE(HasTpuDeviceNodes(fake.dev_dir(), fake.dev_borg_dir(),
                                 fake.proc_devices()));
}

TEST(HasTpuDeviceNodesTest, ReturnsTrueForAccelDirNodeIfProcDevicesUnreadable) {
  ScopedFakeHardwareDir fake;
  // No fake /proc/devices file.
  fake.AddDevAccelDirNodeLinkTo("accel0", "/dev/null");
  EXPECT_TRUE(HasTpuDeviceNodes(fake.dev_dir(), fake.dev_borg_dir(),
                                fake.proc_devices()));
}

TEST(HasTpuDeviceNodesTest, ReturnsTrueWhenDevDirCannotBeListed) {
  if (geteuid() == 0) {
    GTEST_SKIP() << "root can list directories without read permission";
  }
  ScopedFakeHardwareDir fake;
  fake.AddDevNode("null");
  ASSERT_EQ(chmod(fake.dev_dir().c_str(), 0), 0);
  // Without listing the directory, the absence of TPU device nodes is unknown.
  EXPECT_TRUE(HasTpuDeviceNodes(fake.dev_dir(), fake.dev_borg_dir(),
                                fake.proc_devices()));
  ASSERT_EQ(chmod(fake.dev_dir().c_str(), 0700), 0);
}

}  // namespace
}  // namespace torch_tpu
