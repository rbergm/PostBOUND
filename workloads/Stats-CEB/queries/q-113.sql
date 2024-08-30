SELECT COUNT(*)
FROM comments AS c,
  postHistory AS ph,
  badges AS b,
  users AS u
WHERE u.Id = b.UserId
  AND u.Id = ph.UserId
  AND u.Id = c.UserId
  AND c.CreationDate <= CAST('2014-09-10 00:33:30' AS timestamp)
  AND u.DownVotes <= 0
  AND u.UpVotes <= 47;
