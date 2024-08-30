SELECT COUNT(*)
FROM badges AS b, users AS u
WHERE b.UserId = u.Id
  AND u.UpVotes >= 0;
