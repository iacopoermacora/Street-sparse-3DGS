import cf_py_importer
import cf_test_helper.io.recording

rds_client = cf_test_helper.io.recording.recording_details_client.GetSecret('pwecmprcrds-api-key-testing')
print(rds_client)