# Description
Describe the purpose of this pull request.
What is the problem that we're trying to fix, and what is the solution about.

For example:
> There is a bug where deserializing an Isotope creates a new entity separate from the original serialized object. This pull request fixes that, so the singleton behavior is maintained in this case.

If this discussion already appears on an issue that this pull request is meant to solve, that's even better! In that case, a short sentence such as the following suffices:
> This pull request solves issue #5 as decided in that discussion. See the discussion therein.

# Tests
Please describe tests that you added or changed to ensure that things behave correctly. This will ensure that we don't regress on this issue in the future.

For example:
> I added a test called `test_serialize_deserialize` that takes a specific Isotope, serializes and deserializes it with `pickle`, and ensures the result is the same object with an `is` assertion.

# Additional information
If there is some important detail about your changes that you think the reviewer should know about, please list it here.

# Checklist before your pull request is ready
- [ ] I tested the code locally, and it passes without new warnings that weren't there before
- [ ] I reviewed my own code
- [ ] My new code is documented and the documentation to code that I changed code is still applicable
- [ ] Any changes I made to public API that may break downstream packages was accepted by the community in the discussion in the Issue Tracker

Please do these things yourself before you post this pull request to save you and others some time.
