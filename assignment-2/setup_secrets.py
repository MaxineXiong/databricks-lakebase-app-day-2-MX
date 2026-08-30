"""
One-time setup script: stores the Lakebase connection URL in a Databricks
secret scope. Run this locally (with the Databricks CLI configured) or from
a notebook - never commit the resulting secret value anywhere.

Usage:
    python setup_secrets.py
"""
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import workspace
import getpass

w = WorkspaceClient()

if not any(scope.name == 'database' for scope in w.secrets.list_scopes()):
    w.secrets.create_scope(scope="database")
    print('Scope `database` successfully created')
else:
    print('Scope `database` already exists')

w.secrets.put_secret(
    scope="database",
    key="lakebase-url",
    string_value=getpass.getpass("Paste your Lakebase URL: ")
)

w.secrets.put_acl(
    scope="database",
    principal="users",
    permission=workspace.AclPermission.READ,
)

