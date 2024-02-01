package com.example.fyp;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.concurrent.TimeUnit;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;


public class uploadActivity extends AppCompatActivity {

    private static final int PICK_IMAGE_REQUEST = 1;
    private static final int REQUEST_CODE = 2;
    private static final int REQUEST_IMAGE_CAPTURE = 3;
    private static final int CAMERA_PERMISSION_REQUEST_CODE = 4;
    private EditText textField_message;
    private Button uploadImageButton;
    private Button takeImageButton;
    private Button button_send_get;
    private TextView textView_response;
    private String url = "http://192.168.100.2:3000"; // Change to your server URL
    private String POST = "POST";
    private String GET = "GET";
    private File photoFile;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_upload);

        // textField_message = findViewById(R.id.txtField_message);
        uploadImageButton = findViewById(R.id.uploadImage);
        takeImageButton = findViewById(R.id.takeImage);
        // button_send_get = findViewById(R.id.button_send_get);
        textView_response = findViewById(R.id.textView_response);

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {
            // Permission is already granted, proceed with your code
        } else {
            // Permission is not granted, request it
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, REQUEST_CODE);
        }

        uploadImageButton.setOnClickListener(view -> {
            // Create an intent to open the image picker
            Intent i = new Intent(Intent.ACTION_PICK, android.provider.MediaStore.Images.Media.INTERNAL_CONTENT_URI);
            startActivityForResult(i, PICK_IMAGE_REQUEST);
        });
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            // Permission is already granted, proceed with your code
        } else {
            // Permission is not granted, request it
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_REQUEST_CODE);
        }
        takeImageButton.setOnClickListener(view -> {
            // Check if the device has a camera
            if (getPackageManager().hasSystemFeature(PackageManager.FEATURE_CAMERA_ANY)) {
                // Create the File where the photo should go
                photoFile = null;
                try {
                    photoFile = createImageFile();
                } catch (IOException ex) {
                    // Error occurred while creating the File
                    Log.e("TakeImageButton", "Error occurred while creating the File", ex);
                    // You may want to return at this point and handle the error
                }
                // Continue only if the File was successfully created
                if (photoFile != null) {
                    Uri photoURI = FileProvider.getUriForFile(this, "com.example.fyp.fileprovider", photoFile);
                    Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI);
                    startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE);
                }
            } else {
                Toast.makeText(this, "No camera available on this device", Toast.LENGTH_SHORT).show();
            }
        });



//        });
    }

    void sendRequest(String type, String method, String paramname, String param, File imageFile) {
        String fullURL = url + "/" + method + (param == null ? "" : "/" + param);
        Request request;

        OkHttpClient client = new OkHttpClient().newBuilder()
                .connectTimeout(10, TimeUnit.SECONDS)
                .readTimeout(10, TimeUnit.SECONDS)
                .writeTimeout(10, TimeUnit.SECONDS).build();

        if (type.equals(POST)) {
            MultipartBody.Builder requestBodyBuilder = new MultipartBody.Builder()
                    .setType(MultipartBody.FORM);

            // Add the image file as a part of the request body
            if (imageFile != null) {
                requestBodyBuilder.addFormDataPart("file", imageFile.getName(),
                        RequestBody.create(MediaType.parse("image/*"), imageFile));
            }

            // Add other parameters if needed
            if (paramname != null && param != null) {
                requestBodyBuilder.addFormDataPart(paramname, param);
            }

            RequestBody requestBody = requestBodyBuilder.build();

            request = new Request.Builder()
                    .url(fullURL)
                    .post(requestBody)
                    .build();
        } else {
            request = new Request.Builder()
                    .url(fullURL)
                    .build();
        }

        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(@NonNull Call call, @NonNull IOException e) {
                e.printStackTrace();
            }

            @Override
            public void onResponse(Call call, final Response response) throws IOException {
                final String responseData = response.body().string();
                uploadActivity.this.runOnUiThread(() -> textView_response.setText(responseData));
                response.body().close(); // Close the response body
            }
        });
    }

    private String getPathFromUri(Uri uri) {
        String path = null;
        String[] projection = {MediaStore.Images.Media.DATA};
        Cursor cursor = getContentResolver().query(uri, projection, null, null, null);
        if (cursor != null && cursor.moveToFirst()) {
            int columnIndex = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATA);
            path = cursor.getString(columnIndex);
            cursor.close();
        }
        return path;
    }
    private File createImageFile() throws IOException {
        // Create an image file name
        Log.d("CreateImageFile", "Creating image file");
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new Date());
        String imageFileName = "JPEG_" + timeStamp + "_";
        File storageDir = getExternalFilesDir(null);
        File imageFile = File.createTempFile(
                imageFileName,  /* prefix */
                ".jpg",         /* suffix */
                storageDir      /* directory */
        );
        return imageFile;
    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == PICK_IMAGE_REQUEST && resultCode == RESULT_OK && data != null && data.getData() != null) {
            // Get the selected image URI
            Uri imageUri = data.getData();

            // Perform the necessary operations with the selected image URI
            // For example, you can convert it to a file path
            String filePath = getPathFromUri(imageUri);

            // Call the sendRequest method with the file path
            sendRequest(POST, "upload", null, null, new File(filePath));
        } else if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
            // The image was captured successfully
            if (photoFile != null) {
                Uri photoUri = FileProvider.getUriForFile(this, "com.example.fyp.fileprovider", photoFile);
                String filePath = photoFile.getAbsolutePath();
                // Call the sendRequest method with the file path
                sendRequest(POST, "upload", null, null, new File(filePath));
            }
        }
    }


}
