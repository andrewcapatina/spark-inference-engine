# Safe-mode diagnostic review and firmware next step

The user ran the prepared root collector. It completed with exit 0 and produced a readable NVIDIA report. Persistent crash storage was successfully copied and is empty. The raw bundle and full device inventory remain private outside Git; only this sanitized review is committed.

The report has no recorded NVIDIA Xid, OOM-kill or kernel-panic message tied to the latest failure. Earlier NVIDIA refcount status 0x56 messages predate the latest two crashes. Boot-time CPU stack traces show SysRq capture initiated by `nvidia-bug-report` after reboot, not a crash-time panic. “Thermal State: Safe” in SMBIOS is not a measurement of the failure moment. Safe mode omitted dynamic nvidia-smi/vulkan and InfiniBand queries. These limits mean the bundle does not establish or rule out a hardware, firmware, driver or sudden memory failure.

## Concrete firmware lead

The report lists EC 3.5.8, confirmed by a read-only fwupd device query as `0x03000508`, and UEFI inventory `R:1.110.13`. NVIDIA's current Founders Edition release table lists EC 3.13.7 and UEFI 1.111.17. The OS OTA record, kernel and driver match the currently published 7.6.0/7.0.0-1019/580.178.04 stack, so the apparent lag is in firmware rather than proof that all software is old. [Official release notes](https://docs.nvidia.com/dgx/dgx-spark/release-notes.html).

The bundle's SMBIOS type0 BIOS string (`5.36_0ACUM018`) differs from its BIOS firmware-inventory field (`5.36_0ACUM027`). Those fields are recorded separately; the mismatch alone is not diagnosed as a bad flash. PD version formatting also differs between inventory and published tables, so the EC/UEFI evidence is the clearer comparison.

An independent report describes log-less spontaneous power-off on the same kernel/driver combination. In that thread, NVIDIA staff stated on November 5 that newer firmware improved sensor readings and advised using current firmware and temperature/fan data. This is a relevant lead, not evidence that our unit has the same defect or that a firmware update guarantees a fix. [NVIDIA forum discussion](https://forums.developer.nvidia.com/t/dgx-spark-log-less-hard-power-off-implausible-acpi-thermal-sensor-spikes-field-diagnostics-pass-case-261014-000313/386556).

The user independently reports a firmware update offered by DGX Dashboard. NVIDIA recommends Dashboard as the tested update path, including firmware management. The next step is to apply the offered official update through Dashboard, follow its instructions and reboot when prompted; afterward, read firmware versions again before choosing a GPU test. No firmware installation, metadata refresh, package installation or reboot was performed by the assistant. A read-only cached-update-list command was interrupted before producing output, and no fwupdmgr process remained. [Official update guide](https://docs.nvidia.com/dgx/dgx-spark/os-and-component-update.html).

## State retained

GPU and larger-workload holds remain in place. No new optimization is selected. Native inference source is unchanged. Full larger-workload validation remains incomplete; there is no firmware-fix or host-stability claim. The next comparison needs a verified post-update version snapshot, followed by an explicitly selected bounded test with the same diagnostics and one-model policy.
